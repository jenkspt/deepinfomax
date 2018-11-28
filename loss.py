import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform

from models import GlobalDIM, LocalDIM, PriorMatch


class DeepInfoMax(nn.Module):
    def __init__(self, 
            alpha, beta, gamma, 
            encoding_size=64, 
            local_feature_shape=(128, 8, 8)):

        super(DeepInfoMax, self).__init__()

        lC, lH, lW = local_feature_shape
        # Don't waste resources if hyperparameters are set to zero
        self.global_disc = GlobalDIM(lC*lH*lW+encoding_size) if alpha > 0 else None
        self.local_disc = LocalDIM(lC+encoding_size) if beta > 0 else None
        self.prior_disc = PriorMatch(encoding_size) if gamma > 0 else None
        
        # Distributions won't move to GPU
        #self.prior = Uniform(torch.cuda.FloatTensor(0), torch.cuda.FloatTensor(1))

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def global_loss(self, y, M, M_prime):
        b, c, lH, lW = M.shape
        # Global mutual information estimation 
        # Encoder and discriminator should minimize this
        positive_example = torch.cat([M.view(b, -1), y], dim=-1)
        negative_example = torch.cat([M_prime.view(b, -1), y], dim=-1)

        positive = self.global_disc(positive_example)
        negative = self.global_disc(negative_example)
        return mi_bce_loss(positive, negative)

    def local_loss(self, y, M, M_prime):
        b, c, lH, lW = M.shape
        # Local mutual information estimation 
        # Encoder and discriminator should minimize this
        _y = y.unsqueeze(-1).unsqueeze(-1)
        _y = _y.expand(-1, -1, lH, lW)

        positive_example = torch.cat([M, _y], dim=1)
        negative_example = torch.cat([M_prime, _y], dim=1)

        positive = self.local_disc(positive_example)
        negative = self.local_disc(negative_example)
        return mi_bce_loss(positive, negative)/(lH*lW)

    def prior_loss(self, y):
        prior_sample = torch.rand_like(y)
        positive = self.prior_disc(prior_sample)
        negative = self.prior_disc(y)

        prior_disc_loss = mi_bce_loss(positive, negative)   
        return prior_disc_loss

        prior_enc_loss = -prior_disc_loss                   # Encoder loss for prior
        return prior_disc_loss, prior_enc_loss

    def forward(self, y, M, M_prime):

        global_loss = 0.
        local_loss = 0.
        prior_disc_loss = prior_enc_loss = 0

        if not self.global_disc is None:
            global_loss = self.alpha * self.global_loss(y, M, M_prime)

        if not self.local_disc is None:
            local_loss = self.beta * self.local_loss(y, M, M_prime)

        if not self.prior_disc is None:
            prior_loss = self.gamma * self.prior_loss(y)    # Discriminator loss

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
    """ 
    y = torch.rand(4,64)
    M = torch.randn(4,128,8,8)
    M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

    mi_estimator = DeepInfoMax(0, .5, 0, 
            local_feature_shape=M.shape[1:])

    global_loss, local_loss, prior_d_loss, prior_e_loss = mi_estimator(y, M, M_prime)
    print(global_loss, local_loss, prior_d_loss, prior_e_loss)
    print(global_loss + local_loss +prior_d_loss)
    """
