import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessEvalCNN(nn.Module):
    def __init__(self, in_channels: int):
        super(ChessEvalCNN, self).__init__()
        
        # Convolutional layers
        # First layer: from in_channels -> 32 feature maps
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second layer: from 32 -> 64 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third layer: from 64 -> 128 feature maps
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # After the convolutions, we will have (batch_size, 128, 8, 8)
        # Flatten this for the fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 3)
        
    def forward(self, x):
        # x shape: (batch_size, N, 8, 8)
        
        # Pass through first conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Pass through second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Pass through third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Flatten the feature maps
        x = x.view(x.size(0), -1)  # (batch_size, 128*8*8)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (batch_size, 3)
        
        # Apply softmax to get probabilities.
        # For training with CrossEntropyLoss, you would typically omit this softmax here 
        # and let the loss handle it. Here we show it explicitly for clarity.
        probs = F.softmax(logits, dim=1)
        return probs

# Example usage:
# Suppose N=12 for a standard set of piece bitboards (6 for White, 6 for Black), 
# or any other number of input channels.
# model = ChessEvalCNN(in_channels=12)
# input_tensor = torch.randn((32, 12, 8, 8))  # batch_size=32
# output = model(input_tensor)  # output shape: (32, 3) probabilities
