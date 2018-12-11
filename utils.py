
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets.cifar import CIFAR10

from math import floor

def get_features(forward_fn, loader, device=torch.device('cpu')):
    """
    Args:
        forward_fn (callable): function for generating features from a batch
            with `forward` function that returns y, M
        loader (DataLoader): Generates batched inputs to the encoder
    Returns:
        features (torch.Tensor), labels (torch.Tensor)
    """
    assert callable(forward_fn)

    with torch.set_grad_enabled(False):
        features_lst = []
        labels_lst = []
        for images, labels in loader:
            images = images.to(device) 

            feat = forward_fn(images)
            features_lst.append(feat.view(feat.size(0), -1))
            labels_lst.append(labels)

        return torch.cat(features_lst, 0), torch.cat(labels_lst, 0)


def get_loaders(dataset_name='cifar10', batch_size=32):
    if dataset_name == 'cifar10':
        transform = Compose([
            ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_set = CIFAR10('data', train=True, transform=transform)
        train_loader = DataLoader(train_set, batch_size=batch_size, 
                shuffle=True, pin_memory=torch.cuda.is_available())

        valid_set = CIFAR10('data', train=False, transform=transform)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, 
                shuffle=False, pin_memory=torch.cuda.is_available())
    else:
        raise ValueError(f'{dataset_name} not implimented')

    return OrderedDict(train=train_loader, valid=valid_loader)


def jsd_mi(positive, negative):
    """ Jenson-Shannon Divergence Mutual Information Estimation
    Eq. 4 from the paper """
    return (-F.softplus(-positive)).mean() - F.softplus(negative).mean()

def mi_bce_loss(positive, negative):
    """ Eq. 4 from the paper is equivalent to binary cross-entropy 
    i.e. minimizing the output of this function is equivalent to maximizing
    the output of jsd_mi(positive, negative)
    """
    real = F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
    fake = F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
    return real + fake

def mi_nce_loss(positive, negative):
    """ Eq. 5 from the paper """
    raise NotImplementedError

def divergence(positive, negative):
    return torch.log(torch.sigmoid(positive)).mean() + torch.log(1-torch.sigmoid(negative)).mean()


def _conv_size(x, k, s, p, d):
    """ size, kernel, stride, padding, dilation """
    return floor((x + (2 * p) - (d * (k-1)) - 1) / s) + 1

def conv_output_size(size, kernel_size, stride=1, padding=0, dilation=1):
    if isinstance(size, int):
        size = (size,)*2

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)*2
    
    if isinstance(stride, int):
        stride = (stride,)*2

    if isinstance(padding, int):
        padding = (padding,)*2

    if isinstance(dilation, int):
        dilation = (dilation,)*2

    h = _conv_size(size[0], kernel_size[0], stride[0], padding[0], dilation[0])
    w = _conv_size(size[1], kernel_size[1], stride[1], padding[1], dilation[1])
    return h, w


if __name__ == "__main__":
    torch.manual_seed(0)

    positive = torch.randn(32)
    negative = torch.randn(32)

    mi = jsd_mi(positive, negative)
    loss = mi_bce_loss(positive, negative)
    div = divergence(positive, negative)
    print(mi, loss, div)
    
    # Verifies that Eq. 4 from the paper is equivalent to binary cross-entropy
    # And GAN divergence
    assert mi == -loss == div
