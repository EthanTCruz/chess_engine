import torch.nn as nn
import torch.nn.functional as F
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict

class ResidualBlock(nn.Module):
   """
   A standard 2-conv residual block:
   Input -> Conv -> BN -> ReLU -> Conv -> BN -> (Add Skip) -> ReLU -> Output
   """
   def __init__(self, in_channels, out_channels, stride=1, downsample=None):
       super(ResidualBlock, self).__init__()
       self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
       self.bn1 = nn.BatchNorm2d(out_channels)
       self.relu = nn.ReLU(inplace=True)
       self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
       self.bn2 = nn.BatchNorm2d(out_channels)
       self.downsample = downsample  # optional 1x1 conv if shapes differ
   def forward(self, x):
       identity = x
# First conv
       out = self.conv1(x)
       out = self.bn1(out)
       out = self.relu(out)
# Second conv
       out = self.conv2(out)
       out = self.bn2(out)
# Apply the downsample (skip) if needed
       if self.downsample is not None:
           identity = self.downsample(x)
       out += identity
       out = self.relu(out)
       return out


class ChessEvalResNet(nn.Module):
   def __init__(self, in_channels=len(sample_bitboard_dict), num_classes=3):
       super(ChessEvalResNet, self).__init__()
# Initial conv
       self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
       self.bn1   = nn.BatchNorm2d(32)
       self.relu  = nn.ReLU(inplace=True)
# Create multiple layers (stacks of residual blocks)
# E.g. layer1: 32 -> 64, layer2: 64 -> 128
       self.layer1 = self._make_layer(32, 64, blocks=2, stride=2)
       self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
# After two layers with stride=2, your 8x8 might go down to 2x2 if stride=2 is used
# so the final shape might be (batch_size, 128, 2, 2).
# Fully connected
       self.fc = nn.Linear(128 * 2 * 2, num_classes)
   def _make_layer(self, in_channels, out_channels, blocks, stride=1):
       """
       Constructs a 'layer' by stacking multiple ResidualBlocks.
       If in_channels != out_channels or stride != 1, we define a downsample
       so that the skip has the correct shape.
       """
       downsample = None
       if stride != 1 or in_channels != out_channels:
           downsample = nn.Sequential(
               nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               nn.BatchNorm2d(out_channels)
           )
       layers = []
# First block in this layer
       layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=downsample))
# Additional blocks (no downsample needed, stride=1)
       for _ in range(1, blocks):
           layers.append(ResidualBlock(out_channels, out_channels))
       return nn.Sequential(*layers)
   def forward(self, x):
# Initial conv
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)
# Residual layers
       x = self.layer1(x)  # shape could become (batch_size, 64, 4, 4) if stride=2 from 8x8
       x = self.layer2(x)  # shape could become (batch_size, 128, 2, 2)
# Flatten
       x = x.view(x.size(0), -1)  # (batch_size, 128*2*2)
# Classifier
       logits = self.fc(x)
       return logits