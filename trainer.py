import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.distributions import Uniform

from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

from sklearn.svm import LinearSVC



from importlib import reload
from linear_classifier import eval_encoder

import models
reload(models)


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

    return {'train':train_loader, 'valid':valid_loader}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    epochs = 40
    num_classes = 10
    batch_size=32
    alpha = 1
    beta = 0
    gamma = 1


    encoder = models.Encoder(
            input_shape=(3,32,32),
            feature_layer=3)

    mi_estimator = models.MIEstimator(
            alpha, beta, gamma, 
            local_feature_shape=encoder.local_feature_shape,
            encoding_size=encoder.encoding_size)

    encoder.to(device)
    mi_estimator.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=1e-4)
    mi_optim = optim.Adam(mi_estimator.parameters(), lr=1e-4)

    loaders = get_loaders('cifar10', batch_size=batch_size)
   

    for epoch in range(1, epochs+1):
        print(f'\nEpoch {epoch}/{epochs}\n' + '-' * 10)
        

        total_mi_loss = total_prior_loss = 0

        for i, (images, labels) in enumerate(loaders['train']):
            images = images.to(device) 
            encoder.train()

            y, M = encoder(images)
            # Rotate along batch dimension to create M_prime
            M_prime = torch.cat([M[1:], M[0].unsqueeze(0)], dim=0).detach()
            
            global_loss, local_loss, prior_loss = mi_estimator(y, M, M_prime)
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
