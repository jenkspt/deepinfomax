
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.distributions import Uniform

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

from importlib import reload

import loss
import models

reload(loss)
reload(models)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size=32
    alpha = 1
    beta = 0
    gamma = 1
    
    cifar_train = CIFAR10('data', train=True, transform=ToTensor())
    train_loader = DataLoader(cifar_train, batch_size=batch_size, 
            shuffle=True, pin_memory=torch.cuda.is_available())


    encoder = models.Encoder(
            input_shape=(3,32,32),
            feature_layer=3)

    mi_estimator = loss.DeepInfoMax(
            alpha, beta, gamma, 
            local_feature_shape=encoder.local_feature_shape,
            encoding_size=encoder.encoding_size)

    encoder.to(device)
    mi_estimator.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=1e-4)
    mi_optim = optim.Adam(mi_estimator.parameters(), lr=1e-4)
   

    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device) 

            #print(images.shape)
            y, M = encoder(images)
            #print(M.shape)
            # Rotate along batch dimension to create M_prime
            M_prime = torch.cat([M[1:], M[0].unsqueeze(0)], dim=0).detach()
            
            global_loss, local_loss, prior_loss = mi_estimator(y, M, M_prime)
            mi_loss = global_loss + local_loss
            
            # Optimize encoder
            enc_optim.zero_grad()
            encoder_loss = mi_loss - prior_loss 
            encoder_loss.backward(retain_graph=True)
            enc_optim.step()

            mi_optim.zero_grad()
            # Optimize mutual information estimator
            mi_est_loss = mi_loss + prior_loss
            mi_est_loss.backward()
            mi_optim.step()
            if i % 100 == 0:
                print(f'Encoder loss: {encoder_loss:.4f}, MI est loss: {mi_est_loss:.4f}, ', end='')
                print(f'\t( MI(G): {-global_loss:.2f}, ', end='')
                print(f'MI(L): {-local_loss:.2f}, ', end='')
                print(f'Prior loss: {prior_loss:.2f} )')

