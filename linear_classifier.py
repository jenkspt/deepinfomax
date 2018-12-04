import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_features(encoder, loader, device=torch.device('cpu')):
    with torch.set_grad_enabled(False):
        encoder.to(device) 
        encoder.eval()
        features_lst = []
        labels_lst = []
        for images, labels in loader:
            images = images.to(device) 

            y, M = encoder(images)
            features_lst.append(y)
            labels_lst.append(labels)

        return torch.cat(features_lst, 0), torch.cat(labels_lst, 0)

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
        train_loader, 
        eval_loader,
        num_classes,
        train_epochs=5,
        batch_size=128,
        device=torch.device('cpu')):

    # Train classifier on encoder features
    train_features, train_labels = get_features(encoder, train_loader, device)

    train_set = TensorDataset(train_features, train_labels)
    train_features_loader = DataLoader(train_set, batch_size, shuffle=True)

    linear_model = nn.Linear(encoder.encoding_size, num_classes)
    linear_model = train_classifier(
            linear_model, 
            train_features_loader,
            train_epochs,
            device)

    # Evaluate classifier
    eval_features, eval_labels = get_features(encoder, eval_loader, device)
    eval_set = TensorDataset(eval_features, eval_labels)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    accuracy = eval_classifier(linear_model, eval_loader, device)
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
    features = get_features(Encoder(feature_layer=2), img_loader, device)

    # Test train_classifier
    linear_model = train_classifier(nn.Linear(64, 10), feat_loader, epochs=1, device=device)
    # Test eval_classifier
    acc = eval_classifier(nn.Linear(64, 10), feat_loader, device)

    # Test eval_encoder
    acc = eval_encoder(
            Encoder(feature_layer=2),
            img_loader,
            img_loader,
            num_classes=10,
            train_epochs=1,
            batch_size=8,
            device=device)
