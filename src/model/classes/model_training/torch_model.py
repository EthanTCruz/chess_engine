import sys
sys.path.append("../")
import torch
import torch.nn as nn

from chess_engine.src.model.config.config import  model_settings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard

from chess_engine.src.model.classes.autoencoder.AE_Dataloader import get_dataloaders

import random

def set_seed(seed=42):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For CUDA
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables auto-tuning for deterministic results


# Define the ModelOperator class
class ModelOperator:
    def __init__(self,model=None,transform=True):
        set_seed()
        self.batch_size = model_settings.DATALOADER_BATCH_SIZE
        self.num_workers = model_settings.NUM_WORKERS
        self.model_path = model_settings.FULL_MODEL_PATH
        self.transform = transform
        if model:
            self.model = model
        else:
            print("No model provided")


    def create_dataloaders(self, num_workers=0):
        train_loader, test_loader, val_loader = get_dataloaders(self.transform )
        dataloaders = {
            "train": train_loader,
            "valid": val_loader,
            "test": test_loader
        }

        return dataloaders

    def train(self, learning_rate=model_settings.LEARNING_RATE, num_epochs=16, num_workers=model_settings.NUM_WORKERS, save_model=True):
        set_seed()
        num_workers = max(num_workers, self.num_workers)
        dataloaders = self.create_dataloaders(num_workers)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model().to(device)  # Assuming 12 bitboards

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir='runs/cnn')

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
                value_output = model(batch_x1)

                # Compute loss only on the value output
                loss = criterion(value_output, batch_labels)

                if train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * batch_x1.size(0)
                # print(f"value_output: {value_output.shape}")
                # print(f"batch_labels: {batch_labels.shape}")
                batch_labels = batch_labels.argmax(dim=1)
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
        self.model = self.model().to(device)  # Assuming 12 bitboards
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