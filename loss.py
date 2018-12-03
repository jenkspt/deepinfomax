import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from models import GlobalDIM, ConcatAndConvDIM, EncodeAndDotDIM, PriorMatch


class DeepInfoMax(nn.Module):
    def __init__(self, 
            alpha, beta, gamma, 
            encoding_size=64, 
            local_feature_shape=(128, 8, 8),
            encode_and_dot=True):

        super(DeepInfoMax, self).__init__()

        args = (encoding_size, local_feature_shape)
        # Don't waste resources if hyperparameters are set to zero
        self.global_disc = GlobalDIM(*args) if alpha > 0 else None

        if encode_and_dot:
            self.local_disc = EncodeAndDotDIM(*args) if beta > 0 else None
        else:
            self.local_disc = ConcatAndConvDIM(*args) if beta > 0 else None

        self.prior_disc = PriorMatch(encoding_size) if gamma > 0 else None

        # Distributions won't move to GPU
        #self.prior = Uniform(torch.cuda.FloatTensor(0), torch.cuda.FloatTensor(1))

        # DIM Hyperparameters
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, y, M, M_prime):
        # Default values if loss isn't used
        global_loss = local_loss = prior_loss = 0.
        
        # Global MI loss
        if not self.global_disc is None:
            positive = self.global_disc(y, M)
            negative = self.global_disc(y, M_prime)
            global_loss = self.alpha * mi_bce_loss(positive, negative)

        # Local MI loss
        if not self.local_disc is None:
            lH, lW = M.shape[2:]
            positive = self.local_disc(y, M)
            negative = self.local_disc(y, M_prime)
            local_loss = self.beta * mi_bce_loss(positive, negative)/(lH*lW)

        # Prior (discriminator) loss
        if not self.prior_disc is None:
            prior_sample = torch.rand_like(y)
            positive = self.prior_disc(prior_sample)
            negative = self.prior_disc(y)
            prior_loss = self.gamma * mi_bce_loss(positive, negative)

        return global_loss, local_loss, prior_loss


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
    y = torch.rand(4,64)
    M = torch.randn(4,128,8,8)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

    mi_estimator = DeepInfoMax(.5, .5, 1, 
            local_feature_shape=M.shape[1:])

    global_loss, local_loss, prior_loss = mi_estimator(y, M, M_prime)
    print(global_loss, local_loss, prior_loss)
