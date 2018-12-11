import numpy as np
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from importlib import reload
from classifier import eval_encoder
from utils import get_loaders

import models

def load_last_checkpoint(encoder, mi_estimator, dir='checkpoints/dim-G'):
    dir = Path(dir)
    checkpoint_dir = sorted(
            filter(lambda p:p.is_dir(), dir.glob('*')),
            key=lambda p:int(p.name))[-1]
    epoch = int(checkpoint_dir.name)
    encoder.load_state_dict(torch.load(list(checkpoint_dir.glob('encoder*.pt'))[0]))
    mi_estimator.load_state_dict(torch.load(list(checkpoint_dir.glob('mi_estimator*.pt'))[0]))
    return epoch + 1

def save_models(encoder, mi_estimator, epoch, dir):
    ckpt_dir = Path(dir) / str(epoch)
    ckpt_dir.mkdir()
    torch.save(encoder.state_dict(), ckpt_dir / 'encoder.pt')
    torch.save(mi_estimator.state_dict(), ckpt_dir / 'mi_estimator.pt')


def get_models(alpha, beta, gamma, feature_layer=2, input_shape=(3,32,32)):
    encoder = models.Encoder(
            input_shape=input_shape,
            feature_layer=feature_layer)

    mi_estimator = models.MIEstimator(
            alpha, beta, gamma, 
            local_feature_shape=encoder.local_feature_shape,
            encoding_size=encoder.encoding_size)

    return encoder, mi_estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Info Max PyTorch')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='Coefficient for global DIM (default: 1)')
    parser.add_argument('-b', '--beta', type=float, default=0,
                        help='Coefficient for local DIM (default: 0)')
    parser.add_argument('-g', '--gamma', type=float, default=1,
                        help='Coefficient prior matching (default: 1)')
    parser.add_argument('-fl', '--feature-layer', type=int, default=3,
                        help='Layer of encoder to use for local feature M (default: 3)')
    parser.add_argument('--model-name', type=str, default='dim-G',
            help='Checkpoint directory name')
    parser.add_argument('--resume', default=False, action='store_true')

    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10
    save_dir = Path('checkpoints') / args.model_name

    # Creates nn.Modules
    encoder, mi_estimator = get_models(
            args.alpha, args.beta, args.gamma, args.feature_layer)

    if args.resume:
        # Loads in-place
        start_epoch = load_last_checkpoint(encoder, mi_estimator, save_dir)
    else:
        start_epoch = 0
        save_dir.mkdir()

    encoder.to(device)
    mi_estimator.to(device)

    enc_optim = optim.Adam(encoder.parameters(), lr=1e-4)
    mi_optim = optim.Adam(mi_estimator.parameters(), lr=1e-4)

    loaders = get_loaders('cifar10', batch_size=args.batch_size)
   
    for epoch in range(start_epoch, start_epoch + args.epochs + 1):
        print(f'\nEpoch {epoch}/{start_epoch + args.epochs}\n' + '-' * 10)

        for phase, loader in loaders.items():
            if phase == 'train':
                encoder.train()
                mi_estimator.train()
            else:
                encoder.eval()
                mi_estimator.eval()
        

            total_mi_loss = total_prior_loss = 0

            for i, (images, labels) in enumerate(loader):
                images = images.to(device) 
                with torch.set_grad_enabled(phase == 'train'):
                    y, M = encoder(images)
                    
                    global_loss, local_loss, prior_loss = mi_estimator(y, M)
                    mi_loss = global_loss + local_loss

                    total_mi_loss += mi_loss
                    total_prior_loss += prior_loss
                    
                    # Optimize encoder
                    enc_optim.zero_grad()
                    encoder_loss = mi_loss - prior_loss 
                    if phase == 'train':
                        encoder_loss.backward(retain_graph=True)
                        enc_optim.step()


                    mi_optim.zero_grad()
                    # Optimize mutual information estimator
                    mi_est_loss = mi_loss + prior_loss
                    if phase == 'train':
                        mi_est_loss.backward()
                        mi_optim.step()

            num_steps = len(loader.dataset) / args.batch_size
            epoch_mi_loss = total_mi_loss / num_steps
            epoch_prior_loss = total_prior_loss / num_steps

            print(f'{phase} MI loss: {epoch_mi_loss:.4f}, ', end='')
            print(f'Prior loss: {epoch_prior_loss:.4f}')

        save_models(encoder, mi_estimator, epoch, save_dir)

        """
        if epoch % 10 == 0:
            eval_model = nn.Sequential(
                    nn.Linear(encoder.encoding_size, 200),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.Linear(200, NUM_CLASSES)
                )
            eval_model.to(device)
            eval_optimizer = optim.Adam(eval_model.parameters(), 1e-4/2)
            acc = eval_encoder(
                    encoder, 
                    loaders,
                    eval_model,
                    eval_optimizer,
                    train_epochs=50,
                    batch_size=64,
                    device=device)

            print(f'2-layer classifier accuracy: {acc:.2f}')
            torch.save(encoder.state_dict(), f'saved_models/encoder-L-{epoch}.pt')
        """
