#!/usr/bin/env python3

import torch
import torch.nn as nn

# Local imports (adjust paths to match your project)
from chess_engine.src.model.classes.autoencoder.AE_DataLoader import get_dataloaders, FlattenTransform
from chess_engine.src.model.classes.models.autoencoder.SingleInputAutoEncoder import SingleInputAutoencoder
from chess_engine.src.model.config.config import ae_settings
from chess_engine.src.model.classes.models.autoencoder.ExtendedAutoencoder import ExtendedAutoencoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class AutoencoderTrainer:
    def __init__(self):
        self.lr = ae_settings.learningRate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epochs = ae_settings.numEpochs
        self.transform = FlattenTransform()

    def _train_epoch(self,model, dataloader, optimizer, criterion, device='cpu'):
        """
        Trains `model` for one epoch on the given `dataloader`.
        Returns the average training loss over all samples in the dataloader.
        """
        model.train()  # set model to training mode
        running_loss = 0.0
        
        for features_batch, _ in tqdm(dataloader):
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

    def _validate_epoch(self,model, dataloader, criterion, device='cpu'):
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

    def train_autoencoder(self,model, train_loader,writer, val_loader, num_epochs=5, lr=1e-3, device='cpu',model_name='autoencoder'):
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
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = self._validate_epoch(model, val_loader, criterion, device)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        writer.close()
        self.save_model(model,optimizer,model_name)

    def get_autoencoder(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder = SingleInputAutoencoder(latent_dim=128).to(device)
        for ld in ae_settings.LatenDims:
            autoencoder = ExtendedAutoencoder(autoencoder,latent_dim=ld).to(device)
        return autoencoder

    def save_model(self,model, optimizer,model_name):
        model_path = f"{ae_settings.modelFilePath}{model_name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")

    def train_encoder(self):
        writer = SummaryWriter(log_dir='runs/autoencoder')

        # Transform that flattens the [13,8,8]+metadata or whatever your data is into [836].
        



        # 1) Get dataloaders
        train_loader, _, val_loader = get_dataloaders(self.transform )

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
                            model_name=f'base_autoencoder')

        # 4) Initialize the frozen model (encoder frozen, new layers on top)
        #    For example, output_dim=3 if you're predicting 3 values
        for ld in ae_settings.LatenDims:
            writer = SummaryWriter(log_dir=f'runs/autoencoder{ld}')
            autoencoder = ExtendedAutoencoder(autoencoder,latent_dim=ld).to(self.device)
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



