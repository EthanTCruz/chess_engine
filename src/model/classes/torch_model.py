import os
import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from chess_engine.src.model.config.config import  model_settings, data_settings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import sample_bitboard_dict


# Redefine NpzDataset to output labels as class indices
class NpzDataset(Dataset):
    def __init__(self, data_directory, transform=None, target_transform=None):
        """
        Custom Dataset for loading data from multiple .npz files.

        Args:
            data_directory (str): Directory containing the .npz files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): Optional transform to be applied
                on the target.
        """
        self.data_directory = data_directory
        self.transform = transform
        self.target_transform = target_transform
        
        self.files = []
        self.file_sample_counts = []
        self.cumulative_counts = [0]  # Start with 0 to correctly index the first file
        self.file_cache = {}
        self.max_cache_size = 20  # Adjust based on available memory
        
        total_samples = 0
        for file_name in sorted(os.listdir(data_directory)):
            if file_name.endswith('.npz'):
                file_path = os.path.join(data_directory, file_name)
                self.files.append(file_path)
                
                # Load only the header to get the number of samples
                with np.load(file_path) as data:
                    n_samples = data['features'].shape[0]
                self.file_sample_counts.append(n_samples)
                total_samples += n_samples
                self.cumulative_counts.append(total_samples)
        
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.total_samples}")
        
        # Find the file index using binary search
        file_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1
        sample_idx = idx - self.cumulative_counts[file_idx]

        file_path = self.files[file_idx]

        # Use caching to avoid reloading the same file
        if file_path in self.file_cache:
            data = self.file_cache[file_path]
        else:
            if len(self.file_cache) >= self.max_cache_size:
                # Remove the first cached file (simple FIFO cache)
                removed_file = next(iter(self.file_cache))
                del self.file_cache[removed_file]
            data = np.load(file_path)
            self.file_cache[file_path] = data

        features = data['features']
        labels = data['labels']

        # Check if sample_idx is within the bounds of the data arrays
        if sample_idx < 0 or sample_idx >= features.shape[0]:
            raise IndexError(f"Sample index {sample_idx} out of bounds for file {file_path} with size {features.shape[0]}")

        feature_sample = features[sample_idx]
        label_sample = labels[sample_idx]

        # Convert features to tensor
        if self.transform:
            feature_sample = self.transform(feature_sample)
        else:
            feature_sample = torch.from_numpy(feature_sample).float()
        
        # Convert labels from one-hot encoding to class indices
        if self.target_transform:
            label_sample = self.target_transform(label_sample)
        else:
            # label_sample is one-hot encoded, convert to class index
            label_sample = torch.from_numpy(label_sample).long()
            label_sample = torch.argmax(label_sample)
        
        return feature_sample, label_sample

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

# Define the ModelOperator class
class ModelOperator:
    def __init__(self):
        self.batch_size = model_settings.DataLoaderBatchSize
        self.num_workers = model_settings.num_workers
        self.model_path = model_settings.torch_model_file

    def create_dataloaders(self, num_workers=0):
        datasets = {
            "train": NpzDataset(data_settings.npzTrainingDirectory),
            "valid": NpzDataset(data_settings.npzValidationDirectory),
            "test": NpzDataset(data_settings.npzTestingDirectory)
        }

        for key, dataset in datasets.items():
            if len(dataset) == 0:
                raise ValueError(f"{key.capitalize()} dataset is empty. Please check the data loading process.")

        return {
            key: DataLoader(dataset, batch_size=self.batch_size, shuffle=(key=="train"), num_workers=num_workers)
            for key, dataset in datasets.items()
        }

    def train(self, learning_rate=0.001, num_epochs=16, num_workers=0, save_model=True):
        num_workers = max(num_workers, self.num_workers)
        dataloaders = self.create_dataloaders(num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AlphaZeroNet().to(device)  # Assuming 12 bitboards

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

                running_loss += loss.item() * batch_x1.size(0)
                correct += self.calculate_accuracy(value_output, batch_labels)
                total += batch_labels.size(0)

                all_preds.append(value_output.argmax(dim=1))
                all_labels.append(batch_labels)

        avg_loss = running_loss / len(dataloader.dataset)
        accuracy = correct / total * 100
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return avg_loss, accuracy, all_preds, all_labels

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaZeroNet(n_bitboards=12).to(device)  # Assuming 12 bitboards
        self.optimizer = optim.Adam(self.model.parameters())

        checkpoint = torch.load(model_path, map_location=device)
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
        # Labels are class indices, no need to apply torch.max
        return (predicted == labels).sum().item()

# Example usage
if __name__ == '__main__':
    operator = ModelOperator()
    operator.train()
