import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from chess_engine.src.model.classes.sqlite.dataloader import SQLAlchemyDataset
from chess_engine.src.model.config.config import Settings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
from chess_engine.src.model.classes.cnn_bb_scorer import calc_shapes
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict
from chess_engine.src.model.classes.sqlite.models import (GamePositions,
                                                          GamePositionRollup,
                                                          TrainGamePositions,
                                                          ValidationGamePositions,
                                                          TestGamePositions)

class AlphaZeroNet(nn.Module):
    def __init__(self, n_bitboards, board_size=8):
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
        value = F.log_softmax(value, dim=1)  # Softmax for win, lose, draw probabilities
        
        return policy, value

class ModelOperator:
    def __init__(self):
        settings = Settings()


        self.batch_size = settings.DataLoaderBatchSize
        self.num_workers = settings.num_workers
        self.model_path = settings.torch_model_file

    def create_dataloaders(self, num_workers=0):

        datasets = {
            "train": SQLAlchemyDataset(TrainGamePositions, self.batch_size),
            "valid": SQLAlchemyDataset(ValidationGamePositions, self.batch_size),
            "test": SQLAlchemyDataset(TestGamePositions, self.batch_size)
        }

        for key, dataset in datasets.items():
            if len(dataset) == 0:
                raise ValueError(f"{key.capitalize()} dataset is empty. Please check the data loading process.")

        return {
            key: DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
            for key, dataset in datasets.items()
        }

    def train(self, learning_rate=0.001, num_epochs=16, num_workers=0, save_model=True):
        num_workers = max(num_workers, self.num_workers)
        dataloaders = self.create_dataloaders(num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlphaZeroNet(n_bitboards=len(sample_bitboard_dict.keys())).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir='runs/experiment_1')

        for epoch in range(num_epochs):
            train_loss, train_acc, *_ = self._run_epoch(model, dataloaders['train'], optimizer, criterion, device, train=True)
            val_loss, val_acc, *_ = self._run_epoch(model, dataloaders['valid'], optimizer, criterion, device)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/validation', val_acc, epoch)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        test_loss, test_acc, test_preds, test_labels = self._run_epoch(model, dataloaders['test'], optimizer, criterion, device)
        
        if save_model:
            self.save_model(model, optimizer)

        self._show_test_results(test_preds, test_labels)



        writer.close()

    def _run_epoch(self, model, dataloader, optimizer, criterion, device, train=False):
        model.train() if train else model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for batch_x1, batch_labels in tqdm(dataloader):
                batch_x1, batch_labels = batch_x1.to(device), batch_labels.to(device)

                if train:
                    optimizer.zero_grad()

                # Unpack policy and value outputs
                policy_output, value_output = model(batch_x1)
                
                # Compute loss only on the value output
                loss = criterion(value_output, batch_labels)

                if train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                correct += self.calculate_accuracy(value_output, batch_labels)
                total += batch_labels.size(0)

                all_preds.append(value_output.argmax(dim=1))
                all_labels.append(batch_labels.argmax(dim=1))

        avg_loss = running_loss / len(dataloader)
        accuracy = correct / total * 100

        return avg_loss, accuracy, torch.cat(all_preds), torch.cat(all_labels)


    def _show_test_results(self, predictions, labels):
        cm = confusion_matrix(labels.cpu(), predictions.cpu())
        ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()

        precision = precision_score(labels.cpu(), predictions.cpu(), average=None)
        recall = recall_score(labels.cpu(), predictions.cpu(), average=None)
        f1 = f1_score(labels.cpu(), predictions.cpu(), average=None)

        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            print(f"Class {i}: Precision: {p:.4f}, Recall: {r:.4f}, F1-Score: {f:.4f}")

    def load_model(self, model_path):
        shapes = calc_shapes(batch_size=self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet(n_bitboards=len(sample_bitboard_dict.keys())).to(device)
        self.optimizer = optim.Adam(self.model.parameters())

        checkpoint = torch.load(model_path,weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def save_model(self, model, optimizer):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    @staticmethod
    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        return (predicted == labels).sum().item()
