
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

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
