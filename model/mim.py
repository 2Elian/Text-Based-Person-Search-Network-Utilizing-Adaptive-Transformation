from model4 import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from .transformer_utils import zeros_, DropPath, Identity
import paddle
import paddle.nn as nn_1

class ADTNET(nn.Module):
    def __init__(self, args, num_classes=11003,num_fpn_levels=3):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.num_fpn_levels = num_fpn_levels
        #Base Model 
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.patch_size = base_cfg['vision_patch_size']

        #logit
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        #FPN VIT
        self.init_fpn(
            embed_dim=self.embed_dim,
            patch_size=self.patch_size, )
        self.norm = Identity()

        #id match
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        #mlm alingment
        if 'mlm' in args.loss_names or 'style' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64, #用vit-b的头是8
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    #task name
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    #mlm transformer
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    #noise attack
    def add_gaussian_noisy(self, img, alpha_scale=1, beta_scale=1):
        b = img.shape[0]
        c = img.shape[2]
        sigma, mu = torch.std_mean(img, dim=1)
        mu_sigma,mu_mu = torch.std_mean(mu,dim=0,keepdim=True)
        sigma_sigma , sigma_mu = torch.std_mean(sigma,dim=0,keepdim=True)
        mu_dis = torch.distributions.Normal(loc=mu_mu,scale=mu_sigma)
        sigma_dis = torch.distributions.Normal(loc=sigma_mu,scale=sigma_sigma)
        alpha = alpha_scale * mu_dis.sample([b]).squeeze(1).to(self.args.device)#[b,512]
        beta = beta_scale * sigma_dis.sample([b]).squeeze(1).to(self.args.device)#[b,512]
        ep_mu = torch.randn(b,c).to(self.args.device)#[b,512]
        ep_sigma = torch.randn(b,c).to(self.args.device)#[b,512]
        style_mu = (mu + ep_mu*alpha).unsqueeze(1)
        style_sigma = (sigma + ep_sigma*beta).unsqueeze(1)
        x = (img-mu.unsqueeze(1))/sigma.unsqueeze(1)
        y = style_sigma*x + style_mu
        return y


    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()


    #fpn
    def init_fpn(self, embed_dim=512, patch_size=16, out_with_norm=False):
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

            self.fpn3 = Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
            )

            self.fpn2 = Identity()

            self.fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))

        if not out_with_norm:
            self.norm = Identity()
        else:
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, batch , cur_epoch=1):
        ret = dict()
        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        
        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
            
        if 'style' in self.current_task:
            v = image_feats[:,1:,:]#删掉cls token
            style_v = self.add_gaussian_noisy(v) #[b,192,512]
            b = int(style_v.shape[0])
            d = int(self.embed_dim)
            hp = int(384/self.patch_size)
            wp = int(128/self.patch_size)
            print(b)
            print(d)
            print(hp)
            print(wp)
            style_v = torch.reshape(
                    torch.transpose(
                        self.norm(style_v), 2,1),
                    [b, d, hp, wp])#[b,512,384/16,128/16]
            fpns = [self.fpn1, self.fpn2, self.fpn3, self.fpn4][
                -self.num_fpn_levels:]#ex : num_fpn_levels=3 取得就是后三个fpn fpn2 fpn3 fpn4
            outputs = []
            for i, m in enumerate(fpns):
                outputs.append(
                    m(style_v))
            style_v_1 = outputs[0]#torch.Size([64, 512, 48, 16])
            hp1,wp1 = style_v_1.shape[2] , style_v_1.shape[3]
            style_v_2 = outputs[1]#torch.Size([64, 512, 24, 8]) 拿它做mim
            style_v_3 = outputs[2]#torch.Size([64, 512, 12, 4])
            hp3,wp3 = style_v_3.shape[2] , style_v_3.shape[3]
            style_v_1 = torch.transpose(torch.reshape(style_v_1,[b,d,hp1*wp1]),2,1)#[b,48*16,512]
            style_v_2 = torch.transpose(torch.reshape(style_v_2,[b,d,hp*wp]),2,1)#[b,24*8,512]
            style_v_3 = torch.transpose(torch.reshape(style_v_3,[b,d,hp3*wp3]),2,1)#[b,12*4,512]
            style1 = torch.mean(style_v_1,dim=1)
            style2 = torch.mean(style_v_2,dim=1)
            style3 = torch.mean(style_v_3,dim=1)
            ret.update({'style_loss': objectives.compute_style(i_feats,style1,style2,style3,logit_scale)})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = ADTNET(args, num_classes,num_fpn_levels=3)
    # covert model to fp16
    convert_weights(model)
    return model
