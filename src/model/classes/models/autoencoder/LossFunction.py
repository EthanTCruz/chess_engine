import torch
import torch.nn as nn
import torch.optim as optim
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict
from chess_engine.src.model.classes.autoencoder.FeatureExtractor import sample_metadata

class ChessBitboardLoss(nn.Module):
   def __init__(self, lambda_weight=0.1):
       super(ChessBitboardLoss, self).__init__()
       self.lambda_weight = lambda_weight  # Weight for sum difference penalty
   def forward(self, x, x_hat):
       mse_loss = nn.functional.mse_loss(x, x_hat)  # Standard reconstruction loss
        # Compute sum of pieces for input and reconstructed board
       sum_x = torch.sum(x, dim=1) + 1e-6  # Avoid division by zero
       sum_x_hat = torch.sum(x_hat, dim=1)
        # Compute sum difference penalty (scaled)
       sum_diff_penalty = torch.mean((sum_x - sum_x_hat) ** 2 / sum_x)
        # Total loss
       total_loss = mse_loss + self.lambda_weight * sum_diff_penalty
       return total_loss
   
