import torch
import numpy as np

def compute_x_std(x):
    """
    计算视觉特征的方差
    :param x: VIT的输出 [B,L,D]
    :return: 通道方差 [B,1,D]
    """
    N = 1/x.shape[0]
    x_mean = torch.mean(x,dim=1,keepdim=True)
    x_std = torch.sqrt(torch.mul(N,torch.sum(torch.pow(x-x_mean,2),dim=1)) + 0.000001)
    return x_std

def compute_uncer(x):
    """

    :param x: 均值或者方差 [B,1,D]
    :return: 均值的方差 或者方差的方差 代表统计量的不确定性 [1,1,D]
    """
    B = 1/x.shape[0]
    mean = torch.mean(x,dim=0,keepdim=True)
    std = torch.sqrt(torch.mul(B,torch.sum(torch.pow(x-mean,2),dim=0)) + 0.0001).unsqueeze(1)
    return std