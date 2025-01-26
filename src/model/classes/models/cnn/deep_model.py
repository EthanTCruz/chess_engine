import torch.nn as nn
import torch.nn.functional as F
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict

class ChessEvalDeepCNN(nn.Module):
   def __init__(self, in_channels: int = len(sample_bitboard_dict)):
       super(ChessEvalDeepCNN, self).__init__()
# Block 1
       self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
       self.bn1   = nn.BatchNorm2d(32)
       self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
       self.bn2   = nn.BatchNorm2d(32)
# Block 2
       self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
       self.bn3   = nn.BatchNorm2d(64)
       self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
       self.bn4   = nn.BatchNorm2d(64)
# Block 3
       self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
       self.bn5   = nn.BatchNorm2d(128)
       self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
       self.bn6   = nn.BatchNorm2d(128)
# FC layers
       self.fc1 = nn.Linear(128 * 8 * 8, 256)
       self.fc2 = nn.Linear(256, 3)
   def forward(self, x):
# Block 1
       x = F.relu(self.bn1(self.conv1(x)))
       x = F.relu(self.bn2(self.conv2(x)))
# Block 2
       x = F.relu(self.bn3(self.conv3(x)))
       x = F.relu(self.bn4(self.conv4(x)))
# Block 3
       x = F.relu(self.bn5(self.conv5(x)))
       x = F.relu(self.bn6(self.conv6(x)))
# Flatten
       x = x.view(x.size(0), -1)
# Fully connected
       x = F.relu(self.fc1(x))
       logits = self.fc2(x)
       return logits