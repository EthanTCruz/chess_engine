import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):

    """

    Uses a pretrained autoencoder's encoder as a frozen feature extractor,

    then adds new (trainable) layers on top for a new task (like in Deep Chess).

    """

    def __init__(self, pretrained_autoencoder, hidden_dim=64, output_dim=1):

        super().__init__()
        self.frozen_encoder = pretrained_autoencoder.encoder

        for param in self.frozen_encoder.parameters():
            param.requires_grad = False


        self.latent_dim = pretrained_autoencoder.latent_dim
        self.extra_layers = nn.Sequential(

            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)

        )



    def forward(self, x):
        with torch.no_grad():
            z = self.frozen_encoder(x)  # shape [B, latent_dim]
        out = self.extra_layers(z)     # shape [B, output_dim]
        return out