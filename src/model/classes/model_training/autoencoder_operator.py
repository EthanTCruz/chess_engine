#!/usr/bin/env python3

import torch
import torch.nn as nn

# Local imports (adjust paths to match your project)
from chess_engine.src.model.classes.autoencoder.AE_Dataloader import get_dataloaders, FlattenTransform
from chess_engine.src.model.classes.models.autoencoder.SingleInputAutoEncoder import SingleInputAutoencoder
from chess_engine.src.model.config.config import ae_settings
from chess_engine.src.model.classes.models.autoencoder.ExtendedAutoencoder import ExtendedAutoencoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from chess_engine.src.model.classes.models.autoencoder.LossFunction import ChessBitboardLoss

class AutoencoderTrainer:
    def __init__(self):
        self.lr = ae_settings.LEARNING_RATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = ae_settings.EPOCHS
        self.transform = FlattenTransform()
        

    def _train_epoch(self, model, dataloader, optimizer, criterion, device='cpu'):
        """
        Trains `model` for one epoch on the given `dataloader` using BCE loss.
        Returns the average training loss over all samples in the dataloader.
        """
        model.train()  # set model to training mode
        running_loss = 0.0

        for features_batch, _ in tqdm(dataloader, desc="Training"):
            features_batch = features_batch.to(device)

            optimizer.zero_grad()
            # Forward pass
            reconstruction = model(features_batch)
            # Compute loss using BCE
            loss = criterion(reconstruction, features_batch)
            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss (weighted by batch size)
            running_loss += loss.item() * features_batch.size(0)

        # Compute average loss over the epoch
        avg_loss = running_loss / len(dataloader.dataset)
        return avg_loss

    def _validate_epoch(self, model, dataloader, bce_criterion, mse_criterion, device='cpu'):
        """
        Validates `model` on the given `dataloader`.
        Computes and returns the average validation losses over all samples using both BCE and MSE.
        """
        model.eval()  # set model to evaluation mode
        running_bce_loss = 0.0
        running_mse_loss = 0.0

        with torch.no_grad():
            for features_batch, _ in tqdm(dataloader, desc="Validating"):
                features_batch = features_batch.to(device)
                reconstruction = model(features_batch)
                # Compute both losses
                loss_bce = bce_criterion(reconstruction, features_batch)
                loss_mse = mse_criterion(reconstruction, features_batch)
                running_bce_loss += loss_bce.item() * features_batch.size(0)
                running_mse_loss += loss_mse.item() * features_batch.size(0)

        avg_bce_loss = running_bce_loss / len(dataloader.dataset)
        avg_mse_loss = running_mse_loss / len(dataloader.dataset)
        return avg_bce_loss, avg_mse_loss

    def train_autoencoder(self, model, train_loader, writer, val_loader, num_epochs=5, lr=1e-3, device='cpu', model_name='autoencoder'):
        """
        Trains an autoencoder on a reconstruction task.

        Args:
            model (nn.Module): The autoencoder to train.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            num_epochs (int): Number of epochs to train.
            lr (float): Learning rate.
            device (str): 'cuda' or 'cpu'.
            model_name (str): Name used for saving the model.
        """
        # Set up the loss functions: BCE for training and both BCE & MSE for validation
        training_criterion = nn.BCELoss()
        validation_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            # Train for one epoch using BCE loss
            train_loss = self._train_epoch(model, train_loader, optimizer, training_criterion, device)
            # Validate and compute both BCE and MSE losses
            val_bce_loss, val_mse_loss = self._validate_epoch(model, val_loader, training_criterion, validation_criterion, device)

            # Log the losses to TensorBoard
            writer.add_scalar('Loss/train_bce', train_loss, epoch)
            writer.add_scalar('Loss/validation_bce', val_bce_loss, epoch)
            writer.add_scalar('Loss/validation_mse', val_mse_loss, epoch)

            # Print the epoch summary
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Train Loss (BCE): {train_loss:.4f}, "
                  f"Validation Loss (BCE): {val_bce_loss:.4f}, "
                  f"Validation Loss (MSE): {val_mse_loss:.4f}")
        writer.close()
        self.save_model(model, optimizer, model_name)

    def get_autoencoder(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder = SingleInputAutoencoder(latent_dim=128).to(device)
        for ld in ae_settings.LATENT_DIMS:
            autoencoder = ExtendedAutoencoder(autoencoder, latent_dim=ld).to(device)
        return autoencoder

    def save_model(self, model, optimizer, model_name):
        model_path = f"{ae_settings.MODEL_FILE_DIR}{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")

    def train_encoder(self):
        writer = SummaryWriter(log_dir='runs/autoencoder')

        # 1) Get dataloaders
        train_loader, _, val_loader = get_dataloaders(self.transform)

        # 2) Initialize the autoencoder
        autoencoder = SingleInputAutoencoder(latent_dim=128).to(self.device)
        
        # 3) Train the autoencoder
        print("Training Autoencoder...")
        self.train_autoencoder(model=autoencoder,
                                 train_loader=train_loader,
                                 val_loader=val_loader,
                                 num_epochs=self.num_epochs,
                                 lr=self.lr,
                                 device=self.device,
                                 writer=writer,
                                 model_name='base_autoencoder')

        # 4) Extend the autoencoder with additional layers for the new head
        for ld in ae_settings.LATENT_DIMS:
            writer = SummaryWriter(log_dir=f'runs/autoencoder{ld}')
            autoencoder = ExtendedAutoencoder(autoencoder, latent_dim=ld).to(self.device)
            # 5) Train the new head
            print(f"Training new head with latent dim {ld}...")
            self.train_autoencoder(model=autoencoder,
                                     train_loader=train_loader,
                                     val_loader=val_loader,
                                     num_epochs=self.num_epochs,
                                     lr=self.lr,
                                     device=self.device,
                                     writer=writer,
                                     model_name=f'autoencoder{ld}')
            
        return autoencoder
