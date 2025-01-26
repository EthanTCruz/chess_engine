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


        self.encoder = get_encoder(model_path=f"{ae_settings.modelFilePath}autoencoder{ae_settings.LatenDims[-1]}.pth")

        for param in self.encoder.parameters():
            param.requires_grad = False


        self.evaluation = nn.Sequential(nn.Linear(ae_settings.LatenDims[-1], 256),  
                                        nn.ReLU(inplace=True),
                                        nn.Linear(256, 128),  
                                        nn.ReLU(inplace=True),
                                        nn.Linear(128, 64),  
                                        nn.ReLU(inplace=True),
                                        nn.Linear(64, 3))


        
    def forward(self, x):
        with torch.no_grad():
            g = self.encoder(x)
        z = self.evaluation(g)
        return z