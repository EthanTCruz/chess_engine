import torch
import torch.nn as nn
import torch.nn.functional as F
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict

class SkipChessEvalCNN(nn.Module):
   def __init__(self, in_channels: int = len(sample_bitboard_dict)):
       super(ChessEvalCNN, self).__init__()
# Convolutional layers
       self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
       self.bn1   = nn.BatchNorm2d(32)
       self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.bn2   = nn.BatchNorm2d(64)
       self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
       self.bn3   = nn.BatchNorm2d(128)
# We want to add a skip connection from 32->128 channels, so we use 1x1 conv
       self.skip_conv = nn.Conv2d(32, 128, kernel_size=1, stride=1, bias=False)
       self.skip_bn   = nn.BatchNorm2d(128)
# Fully connected layers
       self.fc1 = nn.Linear(128 * 8 * 8, 256)
       self.fc2 = nn.Linear(256, 3)
   def forward(self, x):
# First conv
       x1 = F.relu(self.bn1(self.conv1(x)))  # shape: (batch_size, 32, 8, 8)
# Second conv
       x2 = F.relu(self.bn2(self.conv2(x1))) # shape: (batch_size, 64, 8, 8)
# Third conv (before adding skip)
       x3 = self.bn3(self.conv3(x2))         # shape: (batch_size, 128, 8, 8)
# Create skip from x1 -> shape to 128 channels
       skip = self.skip_bn(self.skip_conv(x1))  # shape: (batch_size, 128, 8, 8)
# Residual addition + final activation
       out = F.relu(x3 + skip)
# Flatten
       out = out.view(out.size(0), -1)  # (batch_size, 128*8*8)
# Fully connected layers
       out = F.relu(self.fc1(out))
       logits = self.fc2(out)  # (batch_size, 3)
       return logits