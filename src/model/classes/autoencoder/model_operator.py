#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
if os.getcwd().endswith('notebooks'):
    print('here')
    os.chdir(r'..')
    os.chdir(r'..')
    os.chdir(r'chess_engine')
import sys
sys.path.append('../')
# Local imports (adjust paths to match your project)
from chess_engine.src.model.classes.autoencoder.AE_DataLoader import get_dataloaders, FlattenTransform
from chess_engine.src.model.classes.autoencoder.SingleInputAutoEncoder import SingleInputAutoencoder
from chess_engine.src.model.config.config import ae_settings
from chess_engine.src.model.classes.autoencoder.ExtendedAutoencoder import ExtendedAutoencoder


def _train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """
    Trains `model` for one epoch on the given `dataloader`.
    Returns the average training loss over all samples in the dataloader.
    """
    model.train()  # set model to training mode
    running_loss = 0.0
    
    for features_batch, _ in dataloader:
        features_batch = features_batch.to(device)

        optimizer.zero_grad()
        # Forward pass
        reconstruction = model(features_batch)
        # Compute loss
        loss = criterion(reconstruction, features_batch)
        # Backprop
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * features_batch.size(0)

    # Average loss = total loss / number of samples
    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def _validate_epoch(model, dataloader, criterion, device='cpu'):
    """
    Validates `model` on the given `dataloader`.
    Returns the average validation loss over all samples in the dataloader.
    """
    model.eval()  # set model to evaluation mode
    running_loss = 0.0

    # No gradient computation needed
    with torch.no_grad():
        for features_batch, _ in dataloader:
            features_batch = features_batch.to(device)
            reconstruction = model(features_batch)
            loss = criterion(reconstruction, features_batch)
            running_loss += loss.item() * features_batch.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def train_autoencoder(model, train_loader, val_loader, num_epochs=5, lr=1e-3, device='cpu'):
    """
    Trains an autoencoder on a reconstruction task.
    
    Args:
        model (nn.Module): The autoencoder to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _validate_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def _train_epoch_new_head(model, dataloader, optimizer, criterion, device='cpu'):
    """
    Trains the new head (frozen encoder + new layers) for one epoch,
    assuming a regression-like task with MSELoss.
    Returns the average training loss.
    """
    model.train()
    running_loss = 0.0

    for features_batch, labels_batch in dataloader:
        features_batch = features_batch.to(device)
        labels_batch = labels_batch.to(device)  # shape [B, 1], for example

        optimizer.zero_grad()
        outputs = model(features_batch)  # shape [B, 1]
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features_batch.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def _validate_epoch_new_head(model, dataloader, criterion, device='cpu'):
    """
    Validates the new head on a regression-like task.
    Returns the average validation loss.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for features_batch, labels_batch in dataloader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device)
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            running_loss += loss.item() * features_batch.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss

def train_new_head(model, train_loader, val_loader, num_epochs=5, lr=1e-3, device='cpu'):
    """
    Trains a 'new head' model (frozen encoder + new MLP layers on top)
    for a regression task using MSELoss.  Adjust as needed for classification.
    
    Args:
        model (nn.Module): The frozen-encoder model with new trainable layers.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        device (str): 'cuda' or 'cpu'.
    """
    criterion = nn.MSELoss()
    # Only update unfrozen parameters in the new head
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr
    )

    for epoch in range(num_epochs):
        train_loss = _train_epoch_new_head(model, train_loader, optimizer, criterion, device)
        val_loss = _validate_epoch_new_head(model, val_loader, criterion, device)

        print(f"NewHead Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def train_encoder():
    """
    Main entry point for training the autoencoder first, then freezing its encoder
    and training a new head on top. This function demonstrates how to set up 
    dataloaders, initialize models, and run the two training phases.
    """
    # Transform that flattens the [13,8,8]+metadata or whatever your data is into [836].
    transform = FlattenTransform()

    # Hyperparameters
    lr = ae_settings.learningRate
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = ae_settings.numEpochs


    # 1) Get dataloaders
    train_loader, _, val_loader = get_dataloaders(transform)

    # 2) Initialize the autoencoder
    autoencoder = SingleInputAutoencoder(input_dim=836, latent_dim=128).to(device)
    
    # 3) Train the autoencoder
    print("Training Autoencoder...")
    train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)

    # 4) Initialize the frozen model (encoder frozen, new layers on top)
    #    For example, output_dim=3 if you're predicting 3 values
    for ld in ae_settings.LatenDims:
        autoencoder = ExtendedAutoencoder(autoencoder,latent_dim=ld)
        # 5) Train the new head
        print(f"Training new head with latent dim {ld}...")
        train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)

    # ex_autoencoder = ExtendedAutoencoder(autoencoder,latent_dim=64)
    # train_autoencoder(ex_autoencoder, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)

    # ex_autoencoder2 = ExtendedAutoencoder(ex_autoencoder,latent_dim=32)
    # train_autoencoder(ex_autoencoder2, train_loader, val_loader, num_epochs=num_epochs, lr=lr, device=device)



    print("Done.")

