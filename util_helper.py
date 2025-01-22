import torch
import numpy as np
# import logging
import random
import os
import torch.distributed as dist
import shutil
import torch.nn.functional as F
import math
from collections.abc import Mapping

def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed ** 2)
    torch.cuda.manual_seed(seed ** 3)
    random.seed(seed ** 4)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def ssim_norm(feature, eps=1e-10):
    
    b,channel,_,_ = feature.shape

    pad_feature = F.pad(feature,(5,5,5,5), mode='reflect')
    feature_ = pad_feature.view(b,channel,-1)
    
    perc_max = feature_.max(2)[0].unsqueeze(2).unsqueeze(2)
    # perc_min = feature_.min(2)[0].unsqueeze(2).unsqueeze(2)
    norm_feature = pad_feature / (perc_max + eps)
    # norm_feature = (pad_feature - perc_min) / (perc_max - perc_min + eps)

    return norm_feature

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter