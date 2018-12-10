import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_classifier(
        model,
        train_loader, 
        epochs=5,
        device=torch.device('cpu')):
    

    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(features)
            _, preds = torch.max(logits, 1)
            loss = ce_loss(logits, labels)

            loss.backward()
            optimizer.step()

    return model

def eval_classifier(
        model,
        valid_loader,
        device=torch.device('cpu')):

    model.to(device)
    model.eval()
    with torch.set_grad_enabled(False):

        running_corrects = 0
        for features, labels in valid_loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            _, preds = torch.max(logits, 1)

            running_corrects += torch.sum(preds == labels.data)
        acc = running_corrects.double() / len(valid_loader.dataset)

    return acc

def eval_encoder(
        encoder,
        loaders,
        eval_model,
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

    train_set = TensorDataset(train_features, train_labels)
    train_features_loader = DataLoader(train_set, batch_size, shuffle=True)

    eval_model = train_classifier(
            eval_model, 
            train_features_loader,
            train_epochs,
            device)

    # Evaluate classifier
    eval_features, eval_labels = get_features(forward_fn, loaders['valid'], device)
    eval_set = TensorDataset(eval_features, eval_labels)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    accuracy = eval_classifier(eval_model, eval_loader, device)
    return accuracy

if __name__ == "__main__":
    from models import Encoder

    # Setup 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_ds = TensorDataset(
            torch.randn(16,3,32,32), 
            torch.randint(0,10, [16], dtype=torch.long))
    img_loader = DataLoader(img_ds)

    feat_ds = TensorDataset(
            torch.randn(16, 64), 
            torch.randint(0,10, [16], dtype=torch.long))
    feat_loader = DataLoader(feat_ds)

    # Test get_features
    encoder = Encoder(feature_layer=2)
    encoder.eval().to(device)
    features = get_features(lambda x: encoder(x)[0], img_loader, device)

    # Test train_classifier
    linear_model = train_classifier(nn.Linear(64, 10), feat_loader, epochs=1, device=device)
    # Test eval_classifier
    acc = eval_classifier(nn.Linear(64, 10), feat_loader, device)

    # Test eval_encoder
    acc = eval_encoder(
            encoder,
            {'train':img_loader, 'valid':img_loader},
            linear_model,
            train_epochs=1,
            batch_size=8,
            device=device)
