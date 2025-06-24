import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import ImageFilter
import random
from .sampler import RandomIdentitySampler
from .sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .REID_DataSet import CUHKPEDES , ICFGPEDES, RSTPReid ,ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}
#这里面写了三个类，但是每个类的初始化参数都没有传，后续调用它的时候再传即可

def build_transforms(img_size=(384,128),aug=False,is_train=True):
    class GaussianBlur(object):
        """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

        def __init__(self, sigma=[.1, 2.]):
            self.sigma = sigma

        def __call__(self, x):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
            return x
        
    height,width = img_size

    mean = [0.38901278, 0.3651612, 0.34836376]
    std = [0.24344306, 0.23738699, 0.23368555]

    if not is_train:
        transform = T.Compose([
            T.Resize((height,width)),
            T.ToTensor(),
            T.Normalize(mean=mean,std=std)
        ])
        return transform
    if aug:
            transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),#以百分之50的概率进行水平翻转
            T.Pad(10),#将图像四周填充10个像素
            T.RandomCrop((height, width)),#随机位置裁剪
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),#以一定的概率对图像进行随机擦除
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform

def collate(batch):#用于DataLoader里面,作为参数传递给它
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)#()后的目的是为指定的类传递初始化参数root
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,#dataset.train 相当于 CUHKPEDES 类里面的self.train 是一个list
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,#我的论文也做MLM任务，所以这个不是特别重要，它只是消融的时候需要用到。
                                     train_transforms,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')#采用分布式的identity采样
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:#使用单GPU的identity采样
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes
