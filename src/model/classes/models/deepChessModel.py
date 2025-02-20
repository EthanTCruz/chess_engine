import torch.nn as nn


from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict
from chess_engine.src.model.classes.autoencoder.FeatureExtractor import sample_metadata
from chess_engine.src.model.classes.models.autoencoder.get_autoencoder_model import get_autoencoder, get_encoder
from chess_engine.src.model.config.config import ae_settings
import torch

input_dim = len(sample_metadata) + len(sample_bitboard_dict)*8*8

class DeepChessModel(nn.Module):
    def __init__(self):
        """
        :param input_dim: Size of flattened input features, e.g., 836
        :param latent_dim: Dimension of the latent space
        """
        super(DeepChessModel, self).__init__()


        self.encoder = get_encoder(model_path=f"{ae_settings.MODEL_FILE_DIR}autoencoder{ae_settings.LATENT_DIMS[-1]}.pth")

        for param in self.encoder.parameters():
            param.requires_grad = False


        # Block 1: latent_dim -> 256
        self.fc1 = nn.Linear(ae_settings.LATENT_DIMS[-1], 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        
        # Block 2: 256 -> 128 with skip connection from Block 1
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # Projection for skip connection (256 -> 128)
        self.skip1 = nn.Linear(256, 128)
        
        # Block 3: 128 -> 64 with skip connection from Block 2
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        # Projection for skip connection (128 -> 64)
        self.skip2 = nn.Linear(128, 64)
        
        # Final layer: 64 -> 3 (e.g., three classes)
        self.fc4 = nn.Linear(64, 3)


        
    def forward(self, x):
        with torch.no_grad():
            g = self.encoder(x)
        # Block 1
        out1 = self.fc1(g)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        
        # Block 2 with skip connection:
        # Process through fc2
        out2 = self.fc2(out1)
        out2 = self.bn2(out2)
        # Compute skip connection from out1 (projected to 128 dims)
        skip_out1 = self.skip1(out1)
        # Add skip connection and apply activation
        out2 = self.relu(out2 + skip_out1)
        
        # Block 3 with skip connection:
        out3 = self.fc3(out2)
        out3 = self.bn3(out3)
        # Compute skip connection from out2 (projected to 64 dims)
        skip_out2 = self.skip2(out2)
        out3 = self.relu(out3 + skip_out2)
        
        # Final evaluation layer
        z = self.fc4(out3)
        return z