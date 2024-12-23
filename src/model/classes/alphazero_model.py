import torch.nn as nn
import torch.nn.functional as F
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict




# Define the AlphaZeroNet model
class AlphaZeroNet(nn.Module):
    def __init__(self, n_bitboards=len(sample_bitboard_dict.keys()), board_size=8):
        super(AlphaZeroNet, self).__init__()
        
        # Input layer: number of channels equals n_bitboards
        self.conv1 = nn.Conv2d(n_bitboards, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Batch normalization layers for each convolution layer
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)  # 2 channels for the policy output
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)   # 1 channel for the value output
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 3)  # 3 outputs for white win, black win, draw

    def forward(self, x):
        # Convolutional layers with ReLU and batch normalization
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)  # Log softmax for policy distribution
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = self.value_fc2(value)
        # Remove log_softmax here, CrossEntropyLoss expects raw logits
        # value = F.log_softmax(value, dim=1)  # Removed
        
        return policy, value