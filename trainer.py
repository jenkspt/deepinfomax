import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.distributions import Uniform

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize



from importlib import reload
from classifier import eval_encoder
from utils import get_loaders

import models
reload(models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Info Max PyTorch')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='Coefficient for global DIM (default: 1)')
    parser.add_argument('-b', '--beta', type=float, default=0,
                        help='Coefficient for local DIM (default: 1)')
    parser.add_argument('-g', '--gamma', type=float, default=0,
                        help='Coefficient prior matching (default: 1)')
    parser.add_argument('-fl', '--feature-layer', type=int, default=3,
                        help='Layer of encoder to use for local feature M (default: 3)')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = models.Encoder(
            input_shape=(3,32,32),
            feature_layer=args.feature_layer)

    mi_estimator = models.MIEstimator(
            args.alpha, args.beta, args.gamma, 
            local_feature_shape=encoder.local_feature_shape,
            encoding_size=encoder.encoding_size)

    encoder.to(device)
    mi_estimator.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=1e-4)
    mi_optim = optim.Adam(mi_estimator.parameters(), lr=1e-4)

    loaders = get_loaders('cifar10', batch_size=args.batch_size)
   

    for epoch in range(1, args.epochs+1):
        print(f'\nEpoch {epoch}/{args.epochs}\n' + '-' * 10)
        

        total_mi_loss = total_prior_loss = 0

        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device) 
            encoder.train()

            y, M = encoder(images)
            
            global_loss, local_loss, prior_loss = mi_estimator(y, M)
            mi_loss = global_loss + local_loss

            total_mi_loss += mi_loss
            total_prior_loss += prior_loss
            
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

        num_steps = len(loaders['train'].dataset) / batch_size
        epoch_mi_loss = total_mi_loss / num_steps
        epoch_prior_loss = total_prior_loss / num_steps
        print(f'MI loss: {epoch_mi_loss:.4f}, ', end='')
        print(f'Prior loss: {epoch_prior_loss:.4f}')
        
        if False: #epoch % 10 != 0:

            # Trains a linear classifier on encodings and then
            # evaluates on test set
            eval_model = nn.Linear(encoder.encoding_size, num_classes)
            acc = eval_encoder(
                    encoder, 
                    loaders,
                    eval_model,
                    train_epochs=5,
                    batch_size=128,
                    device=device)

            print(f'Linear classifier accuracy: {acc:.2f}')

        if epoch % 10 == 0:
            eval_model = nn.Sequential(
                    nn.Linear(encoder.encoding_size, 200),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(200, num_classes)
                )
            acc = eval_encoder(
                    encoder, 
                    loaders,
                    eval_model,
                    train_epochs=100,
                    batch_size=64,
                    device=device)

            print(f'2-layer classifier accuracy: {acc:.2f}')
