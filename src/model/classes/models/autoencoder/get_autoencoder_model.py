from chess_engine.src.model.classes.models.autoencoder.SingleInputAutoEncoder import SingleInputAutoencoder
from chess_engine.src.model.config.config import ae_settings
from chess_engine.src.model.classes.models.autoencoder.ExtendedAutoencoder import ExtendedAutoencoder
import torch
import os



def get_autoencoder(selected_encoder: int = ae_settings.LatenDims[-1]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = SingleInputAutoencoder(latent_dim=128).to(device)
    for ld in ae_settings.LatenDims:
        autoencoder = ExtendedAutoencoder(autoencoder,latent_dim=ld).to(device)
        if ld == selected_encoder:
            break
    return autoencoder

def get_encoder( model_path: str = f"{ae_settings.modelFilePath}autoencoder{ae_settings.LatenDims[-1]}.pth"):
    if os.path.exists(model_path) == False:
        print(f"Model not found at {model_path}")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    auto_encoder = get_autoencoder()


    optimizer = torch.optim.Adam(auto_encoder.parameters())




    checkpoint = torch.load(model_path, map_location=device)
    auto_encoder.load_state_dict(checkpoint['model_state_dict'])
    
    encoder = auto_encoder.get_encoder()
    encoder = encoder.to(device)
    encoder.eval()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Model loaded from {model_path}")

    return encoder