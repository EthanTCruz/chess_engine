import torch.nn as nn


from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict
from chess_engine.src.model.classes.autoencoder.FeatureExtractor import sample_metadata

input_dim = len(sample_metadata) + len(sample_bitboard_dict)*8*8



class SingleInputAutoencoder(nn.Module):
    def __init__(self, input_dim=input_dim, latent_dim=128):
        """
        :param input_dim: Size of flattened input features, e.g., 836
        :param latent_dim: Dimension of the latent space
        """
        super(SingleInputAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(latent_dim)

            # Optionally add an activation or not, depending on how you want your latent space
        )

        # Decoder
        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, input_dim),
            # For final activation, if your data is normalized [0,1] you might do Sigmoid here
            nn.Sigmoid()
        )

    def encode(self, x):
       return self.encoder(x)
    
    def get_encoder(self):

                               
       return self.encoder
    
    def get_decoder(self):

       return self.decoder

    def decode(self, z):
       return self.decoder(z)
        
    def forward(self, x):
        """
        x shape: [B, 836]
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon