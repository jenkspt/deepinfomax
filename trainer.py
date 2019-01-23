import numpy as np
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboard import Tensorboard

from importlib import reload
#from classifier import eval_encoder
from utils import get_loaders

import models

def load_checkpoint(models, save_dir, epoch=None):
    """
    Args:
        models (dict[str, nn.Module]): Subset of models used in save_checkpoint
        save_dir (str or Path): Parent directory where checkpoints are saved
        epoch (int): Epoch of checkpoit to load. If not provided, loads latest checkpoint
    """
    
    key = lambda p: int(p.name.split('_')[1].split('-')[1])  # Get epoch from filename
    ckpt_paths = {key(p):p for p in save_dir.glob('ckpt*.tar')}

    epoch = epoch if epoch else sorted(ckpt_paths.keys())[-1]
    ckpt_path = ckpt_paths[epoch]

    ckpt = torch.load(ckpt_path)

    for key, model in models.items():
        model.load_state_dict(ckpt[key])

    return epoch


def save_checkpoint(models, save_dir, epoch, loss):
    """
    Args:
        models (dict[str, nn.Module]): Dictionary of models to save
        save_dir (Path): Directory to save checkpoint in
        epoch (int): Current epoch
        loss (float): Current loss
    """

    save_dir = Path(save_dir)
    assert save_dir.is_dir()

    torch.save({
        **{key:model.state_dict() for key, model in models.items()},
        'epoch': epoch,
        'loss': loss}, save_dir / f'ckpt_epoch-{epoch}_loss-{loss:.4f}.tar')


def get_dim(alpha, beta, gamma, feature_layer=2, input_shape=(3,32,32), num_negative=2):

    encoder = models.Encoder(
            input_shape=input_shape,
            feature_layer=feature_layer)

    dim = models.DIM(alpha, beta, gamma, encoder, num_negative=num_negative)
    return dim

def remove_checkpoints(save_dir, keep_num=5):
    save_dir = Path(save_dir)
    key = lambda p: int(p.name.split('_')[1].split('-')[1])  # Get epoch from filename
    ckpts = sorted(save_dir.glob('ckpt*.tar'), key=key)[:-keep_num]
    for ckpt in ckpts:
        os.remove(ckpt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Info Max PyTorch')
    parser.add_argument('--batch-size', type=int, default=32+16,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num-negative', type=int, default=1,
                        help='Number of negative examples to use per positive (default: 1)')
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
    print(args)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_CLASSES = 10

    save_dir = Path('checkpoints') / args.model_name


    # Creates nn.Modules
    dim = get_dim(
            args.alpha, 
            args.beta, 
            args.gamma, 
            args.feature_layer, 
            num_negative=args.num_negative)
    
    dim.to(device)
    
    mi_params = list(dim.encoder.parameters())
    mi_params += list(dim.global_disc.parameters()) if dim.global_disc else []
    mi_params += list(dim.local_disc.parameters()) if dim.local_disc else []
    mi_optim = optim.Adam(mi_params, args.lr)

    prior_optim = optim.Adam(dim.prior_disc.parameters(), args.lr)

    models = {
            'dim': dim,
            'mi_optimizer': mi_optim,
            'prior_optimizer': prior_optim}

    if args.resume:
        # Loads in-place
        start_epoch = load_checkpoint(models, save_dir)
    else:
        start_epoch = 0
        save_dir.mkdir()

    loggers = {'train':Tensorboard(str(save_dir / 'train')), 
            'valid':Tensorboard(str(save_dir / 'valid'))}


    loaders = get_loaders('cifar10', batch_size=args.batch_size)
   
    for epoch in range(start_epoch+1, start_epoch + args.epochs+1):
        print(f'\nEpoch {epoch}/{start_epoch + args.epochs}\n' + '-' * 10)

        for phase, loader in loaders.items():
            if phase == 'train':
                dim.train()
            else:
                dim.eval()

            total_mi_loss = total_prior_loss = 0

            for images, labels in loader:
                images = images.to(device) 
                with torch.set_grad_enabled(phase == 'train'):

                    mi_optim.zero_grad()
                    prior_optim.zero_grad()

                    mi_loss, prior_loss = dim(images)
                    # Optimize encoder
                    if phase == 'train':

                        (mi_loss - prior_loss).backward(retain_graph=True)
                        mi_optim.step()

                        prior_loss.backward()
                        prior_optim.step()

                    total_mi_loss += mi_loss.item()
                    total_prior_loss += prior_loss.item()


            num_steps = len(loader.dataset) / args.batch_size
            epoch_mi_loss = total_mi_loss / num_steps
            epoch_prior_loss = total_prior_loss / num_steps

            print(f'{phase} MI loss: {epoch_mi_loss:.5f}, ', end='')
            print(f'Prior loss: {epoch_prior_loss:.5f}')

            loggers[phase].log_scalar('MI-loss', epoch_mi_loss, epoch)
            loggers[phase].log_scalar('Prior-loss', prior_loss, epoch)

        save_checkpoint(models, save_dir, epoch, mi_loss)
        remove_checkpoints(save_dir, keep_num=10)
