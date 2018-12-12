from collections import OrderedDict
from pathlib import Path
import copy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils import get_features
from trainer import get_models, load_last_checkpoint

def train_model(model, loaders, optimizer, num_epochs, device=torch.device('cpu')):

    criterion = nn.CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float('-inf')

    for epoch in range(1,num_epochs+1):
        print(f'\nEpoch {epoch}/{num_epochs}' + '-' * 10)

        # Each epoch has a training and validation phase
        for phase, loader in loaders.items():
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset) 

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase != 'train' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_encoder(
        encoder,
        loaders,
        eval_model,
        optimizer,
        train_epochs=5,
        batch_size=128,
        feature_layer='y',
        device=torch.device('cpu')):

    """
    Args:
        encoder (nn.Module): Encoder function to generate features from,
            with `forward` function that returns y, M
        loaders (dict(DataLoader)): dictionary with keys 'valid' and 'train',
            which contain DataLoaders
        eval_model (nn.Module): Module to train on encoder features
        train_epochs (int): Number of training epochs to train eval_model
        batch_size  (int): Batch size for training eval_model
        feature_layer (str): One of 'y', 'fc', 'conv' (case insensitive). The
            layer in the encoder from which to generate features
        device (torch.device): 'cpu' or 'cuda'
    Returns
        (int): Accuracy of the eval_model trained on encoder features
    """
    encoder.eval()
    feature_layer = feature_layer.lower()
    if feature_layer == 'y':
        forward_fn = lambda images: encoder(images)[0]
    elif feature_layer == 'fc':
        forward_fn = lambda images: encoder.f1(encoder.f0(encoder.C(images)))
    elif feature_layer == 'conv':
        # Encoder needs to implement an interface
        forward_fn = lambda images: encoder.f0(encoder.C(images))
    else:
        raise ValueError(f"feature_layer was {feature_layer}" 
                + "but should be one of 'y', 'fc', 'conv'")

    # Train classifier on encoder features
    train_features, train_labels = get_features(forward_fn, loaders['train'], device)
    eval_features, eval_labels = get_features(forward_fn, loaders['valid'], device)
    encoder.to('cpu')

    train_set = TensorDataset(train_features, train_labels)
    eval_set = TensorDataset(eval_features, eval_labels)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    loaders = OrderedDict(train=train_loader, valid=eval_loader)

    eval_model = train_model(eval_model, loaders, optimizer, train_epochs, device)


if __name__ == "__main__":
    from models import Encoder
    from utils import get_loaders

    # Setup 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    epochs = 10
    alpha, beta, gamma = 0, 1, .1
    feature_layer = 2
    save_dir = Path('checkpoints/dim-L')
    
    encoder, mi_estimator = get_models(alpha, beta, gamma, feature_layer)
    start_epoch = load_last_checkpoint(encoder, mi_estimator, save_dir)

    loaders = get_loaders('cifar10', batch_size=128)

    encoder.to(device)
   
    eval_model = nn.Sequential(
            nn.Linear(encoder.encoding_size, 200),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(200, 10)
        )

    eval_model.to(device)

    optimizer = optim.Adam(eval_model.parameters(), 1e-4)

    eval_encoder(
            encoder, 
            loaders,
            eval_model,
            optimizer,
            train_epochs=100,
            batch_size=128,
            device=device)
