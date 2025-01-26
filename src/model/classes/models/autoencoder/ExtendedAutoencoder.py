from torch import nn
import torch

class ExtendedAutoencoder(nn.Module):
    def __init__(self, trained_autoencoder, latent_dim=64):
        super().__init__()
        
        # The frozen encoder
        self.encoder = trained_autoencoder.get_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = False


        self.latent_dim = latent_dim

        
        
        
        self.enc_latent_layers = nn.Sequential(nn.Linear(trained_autoencoder.latent_dim, self.latent_dim ),
                                     nn.LeakyReLU(inplace=True),)


        self.dec_latent_layers = nn.Sequential(nn.Linear(self.latent_dim , trained_autoencoder.latent_dim),
                                                nn.LeakyReLU(inplace=True),)
        # Reuse the old decoder
        self.decoder = trained_autoencoder.get_decoder()



    def get_encoder(self):
       encoder = nn.Sequential(self.encoder,
                               self.enc_latent_layers)
       return encoder
    

    def get_decoder(self):
       decoder = nn.Sequential(self.dec_latent_layers,
                               self.decoder)
       return decoder

    def encode(self, x):
       return self.encoder(x)

    def decode(self, z):
       return self.decoder(z)
        
    def forward(self, x):
        # Pass through frozen encoder
        with torch.no_grad():
            z = self.encoder(x)
        z = self.enc_latent_layers(z)
        
        # Pass into the old decoder
        z = self.dec_latent_layers(z)
        x_recon = self.decoder(z)
        return x_recon
