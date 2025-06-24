import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    """
    tensor([[   0,   -1,   -1,    0],
            [  -1,    0,    0,   -1],
            [   0,   -1,    0,   -1],
            [-100,   -1,    0,    0]])
    """
    labels = (pid_dist == 0).float()
    """
    tensor([[1., 0., 0., 1.],
            [0., 1., 1., 0.],
            [1., 0., 1., 0.],
            [0., 0., 1., 1.]])
    """

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) #dim=1 是对行求

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    image_norm = image_norm.float()
    text_norm = text_norm.float()
    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

def compute_style(image_features, style1, style2 , style3 , style_v,logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    style_norm = style_v / style_v.norm(dim=-1,keepdim=True)
    style1_norm = style1 / style1.norm(dim=-1, keepdim=True)
    style2_norm = style2 / style2.norm(dim=-1, keepdim=True)
    style3_norm = style3 / style3.norm(dim=-1, keepdim=True)
    style1_norm = style1_norm.float()
    image_norm = image_norm.float()
    style_norm = style_norm.float()
    style2_norm = style2_norm.float()
    style3_norm = style3_norm.float()

    # cosine similarity as logits
    logits_per_image1 = logit_scale * image_norm @ style_norm.t()
    logits_per_image2 = logit_scale * style1_norm @ style_norm.t()
    logits_per_image3 = logit_scale * style2_norm @ style_norm.t()
    logits_per_image4 = logit_scale * style3_norm @ style_norm.t()


    loss_i1 = F.cross_entropy(logits_per_image1, labels)
    loss_i2 = F.cross_entropy(logits_per_image2, labels)
    loss_i3 = F.cross_entropy(logits_per_image3, labels)
    loss_i4 = F.cross_entropy(logits_per_image4, labels)
    
    loss = loss_i1 + 0.5 * loss_i2 + 0.5 * loss_i3 + 0.5 * loss_i4

    return loss

def compute_style1(image_features, style1, style2 , style3 , style_v,logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    style_norm = style_v / style_v.norm(dim=-1,keepdim=True)
    style1_norm = style1 / style1.norm(dim=-1, keepdim=True)
    style2_norm = style2 / style2.norm(dim=-1, keepdim=True)
    style3_norm = style3 / style3.norm(dim=-1, keepdim=True)
    style1_norm = style1_norm.float()
    image_norm = image_norm.float()
    style_norm = style_norm.float()
    style2_norm = style2_norm.float()
    style3_norm = style3_norm.float()

    # cosine similarity as logits
    logits_per_image1 = logit_scale * image_norm @ style_norm.t()
    logits_per_image2 = logit_scale * style1_norm @ style_norm.t()
    logits_per_image3 = logit_scale * style2_norm @ style_norm.t()
    logits_per_image4 = logit_scale * style3_norm @ style_norm.t()


    loss_i1 = F.cross_entropy(logits_per_image1, labels)
    loss_i2 = F.cross_entropy(logits_per_image2, labels)
    loss_i3 = F.cross_entropy(logits_per_image3, labels)
    loss_i4 = F.cross_entropy(logits_per_image4, labels)
    
    loss = loss_i1 + loss_i2 +  loss_i3 +  loss_i4

    return loss

def compute_style1_1(image_features, style1,  style3 , style_v,logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    style_norm = style_v / style_v.norm(dim=-1,keepdim=True)
    style1_norm = style1 / style1.norm(dim=-1, keepdim=True)
    style3_norm = style3 / style3.norm(dim=-1, keepdim=True)
    style1_norm = style1_norm.float()
    image_norm = image_norm.float()
    style_norm = style_norm.float()
    style3_norm = style3_norm.float()

    # cosine similarity as logits
    logits_per_image1 = logit_scale * image_norm @ style_norm.t()
    logits_per_image2 = logit_scale * style1_norm @ style_norm.t()
    logits_per_image4 = logit_scale * style3_norm @ style_norm.t()


    loss_i1 = F.cross_entropy(logits_per_image1, labels)
    loss_i2 = F.cross_entropy(logits_per_image2, labels)
    loss_i4 = F.cross_entropy(logits_per_image4, labels)
    
    loss = loss_i1 + 0.5 * loss_i2 + 0.5 * loss_i4

    return loss
def compute_id(image_logits, text_logits, labels):
    """
    Instance loss proposed at http://arxiv.org/abs/1711.05535 #大改
    """
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def compute_dma(image_fetures, text_fetures, pid, logit_scale, sigma , tao , image_id=None, factor=0.3, epsilon=1e-8):
    """
    Adaptive Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    labels_distribute = labels / labels.sum(dim=1)
    print('label:{}'.format(labels_distribute))
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    t2i_dis = t2i_cosine_theta
    i2t_cosine_theta = t2i_cosine_theta.t()
    i2t_dis = i2t_cosine_theta

    t2i_scaling_matrix = 1/(torch.exp(-t2i_dis / (2 * sigma ** 2))*tao)#0.5sigma , 0.035_tao
    print(t2i_scaling_matrix)
    i2t_scaling_matrix = 1/(torch.exp(-i2t_dis / (2 * sigma ** 2))*tao)

    # print('未乘原型矩阵之前：{}'.format(t2i_cosine_theta))
    # print('Gauss_dis:{}'.format(text_proj_image))
    # t2i_mean = t2i_cosine_theta.mean(dim=1)
    # t2i_std = t2i_cosine_theta.std(dim=1)
    # t2i_scaling_matrix = 1/(torch.sigmoid(torch.nn.Parameter(
    #     torch.empty((t2i_cosine_theta.size(0), t2i_cosine_theta.size(0))).normal_(mean=t2i_mean.view(-1, 1),
    #                                                                               std=t2i_std.view(-1, 1)))))
    # i2t_mean = i2t_cosine_theta.mean(dim=1)
    # i2t_std = i2t_cosine_theta.std(dim=1)
    # i2t_scaling_matrix = 1/(torch.sigmoid(torch.nn.Parameter(
    #     torch.empty((t2i_cosine_theta.size(0), t2i_cosine_theta.size(0))).normal_(mean=i2t_mean.view(-1, 1),
    #                                                                               std=i2t_std.view(-1, 1)))))

    text_proj_image = t2i_scaling_matrix * t2i_cosine_theta
    image_proj_text = i2t_scaling_matrix * i2t_cosine_theta

    # normalize the true matching distribution
    

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply

def get_itc_style(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    
    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    image_norm = image_norm.float()
    text_norm = text_norm.float()
    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()

    loss = F.cross_entropy(logits_per_image, labels)

    return loss
def get_lu_loss(raw_image_feats,scale1_feats,scale2_feats, style_feats,temp,total_epoch, cur_epoch):
    se = get_itc_style(raw_image_feats, style_feats,temp)
    std = torch.std(raw_image_feats)
    inv_std = torch.exp(-std)

    mse = torch.mean(inv_std * se)
    reg = torch.mean(std)
    L_u = (mse + reg) / 2
    L_1 = get_itc_style(scale1_feats, style_feats,temp)
    L_2 = get_itc_style(scale2_feats, style_feats,temp)
    L_info = 0.5*L_1 + 0.5*(L_2 + L_1)
    gamma = np.exp(- cur_epoch / total_epoch)
    return gamma * L_u + (1 - gamma) * L_info

def get_lu_loss_1(raw_image_feats,scale1_feats,scale2_feats, style_feats,temp,total_epoch, cur_epoch):
    L_u = get_itc_style(raw_image_feats, style_feats,temp)

    L_1 = get_itc_style(scale1_feats, style_feats,temp)
    L_2 = get_itc_style(scale2_feats, style_feats,temp)
    L_info = 0.5*L_1 + 0.5*(L_2 + L_1)
    gamma = np.exp(- cur_epoch / total_epoch)
    return gamma * L_u + (1 - gamma) * L_info

def bi_sdm(image_fetures, text_fetures, pid, logit_scale, total_epoch , cur_epoch, image_id=None, factor=0.3, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    """
    tensor([[   0,   -1,   -1,    0],
            [  -1,    0,    0,   -1],
            [   0,   -1,    0,   -1],
            [-100,   -1,    0,    0]])
    """
    labels = (pid_dist == 0).float()
    """
    tensor([[1., 0., 0., 1.],
            [0., 1., 1., 0.],
            [1., 0., 1., 0.],
            [0., 0., 1., 1.]])
    """

    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1) #dim=1 是对行求

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    q2p_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    p2q_i2t_pred = labels_distribute * (torch.log(labels_distribute + epsilon) - F.log_softmax(image_proj_text,dim=1) )
    p2q_t2i_pred = labels_distribute * (torch.log(labels_distribute + epsilon) - F.log_softmax(text_proj_image,dim=1) )
    p2q_loss = torch.mean(torch.sum(p2q_i2t_pred,dim=1)) + torch.mean(torch.sum(p2q_t2i_pred,dim=1))

    gama = np.exp(-cur_epoch/total_epoch)

    loss = gama * q2p_loss + (1-gama) * p2q_loss

    return loss

def compute_asdm(image_fetures, text_fetures, pid, logit_scale, sigma , tao , image_id=None, factor=0.3, epsilon=1e-8):
    """
    Adaptive Similarity Distribution Matching
    """
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()
    labels_distribute = labels / labels.sum(dim=1)
    # print('label:{}'.format(labels_distribute))
    if image_id != None:
        # print("Mix PID and ImageID to create soft label.")
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask
        # labels = (labels + image_id_mask) / 2

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    t2i_dis = t2i_cosine_theta
    i2t_cosine_theta = t2i_cosine_theta.t()
    i2t_dis = i2t_cosine_theta

    t2i_scaling_matrix = 1/(torch.exp(-t2i_dis / (2 * sigma ** 2))*tao)#0.5sigma , 0.035_tao
    # print(t2i_scaling_matrix)
    i2t_scaling_matrix = 1/(torch.exp(-i2t_dis / (2 * sigma ** 2))*tao)

    # print('未乘原型矩阵之前：{}'.format(t2i_cosine_theta))
    # print('Gauss_dis:{}'.format(text_proj_image))
    # t2i_mean = t2i_cosine_theta.mean(dim=1)
    # t2i_std = t2i_cosine_theta.std(dim=1)
    # t2i_scaling_matrix = 1/(torch.sigmoid(torch.nn.Parameter(
    #     torch.empty((t2i_cosine_theta.size(0), t2i_cosine_theta.size(0))).normal_(mean=t2i_mean.view(-1, 1),
    #                                                                               std=t2i_std.view(-1, 1)))))
    # i2t_mean = i2t_cosine_theta.mean(dim=1)
    # i2t_std = i2t_cosine_theta.std(dim=1)
    # i2t_scaling_matrix = 1/(torch.sigmoid(torch.nn.Parameter(
    #     torch.empty((t2i_cosine_theta.size(0), t2i_cosine_theta.size(0))).normal_(mean=i2t_mean.view(-1, 1),
    #                                                                               std=i2t_std.view(-1, 1)))))

    text_proj_image = t2i_scaling_matrix * t2i_cosine_theta
    image_proj_text = i2t_scaling_matrix * i2t_cosine_theta

    # normalize the true matching distribution
    

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss
