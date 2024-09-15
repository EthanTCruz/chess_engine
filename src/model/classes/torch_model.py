import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from chess_engine.src.model.classes.MongoDBDataset import MongoDBDataset
from chess_engine.src.model.config.config import Settings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard
from chess_engine.src.model.classes.cnn_bb_scorer import calc_shapes
from pymongo import MongoClient

class FullModel(nn.Module):
    def __init__(self, input_planes, additional_features, output_classes=3):
        super(FullModel, self).__init__()




        self.conv1 = nn.Conv2d(in_channels=input_planes, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(256 * 8 * 8 + 64 * 8 * 8, 1024)  # Adjust input dimension to match concatenated features
        self.fc2 = nn.Linear(1024, output_classes)

        self.fc_additional = nn.Linear(additional_features, 64 * 8 * 8)
        
    def forward(self, bitboards, metadata):
        # print(f"After start shape: {bitboards.shape}")
        x = F.relu(self.conv1(bitboards))
        # print(f"After conv1: {x.shape}")

        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")

        x = F.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")

        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f"After flatten: {x.shape}")

        # Process metadata through fc_additional
        metadata_processed = F.relu(self.fc_additional(metadata))
        # print(f"After fc_additional: {metadata_processed.shape}")

        # Flatten metadata_processed
        metadata_processed = metadata_processed.view(metadata_processed.size(0), -1)
        # print(f"After flatten metadata_processed: {metadata_processed.shape}")

        # Concatenate bitboards and metadata features
        combined_features = torch.cat((x, metadata_processed), dim=1)
        # print(f"After concatenation: {combined_features.shape}")

        x = F.relu(self.fc1(combined_features))
        # print(f"After fc1: {x.shape}")

        x = self.fc2(x)
        # print(f"After fc2: {x.shape}")

        x = F.log_softmax(x, dim=1)  # Softmax output
        # print(f"After softmax: {x.shape}")

        return x


class ModelOperator:
    def __init__(self):
        settings = Settings()
        self.train_collection = settings.training_collection_key
        self.test_collection = settings.testing_collection_key
        self.valid_collection = settings.validation_collection_key
        self.mongo_url = settings.mongo_url
        self.db_name = settings.db_name
        self.batch_size = settings.DataLoaderBatchSize
        self.num_workers = settings.num_workers
        self.model_path = settings.torch_model_file

    def create_dataloaders(self, num_workers=0):
        client =  MongoClient(self.mongo_url, maxPoolSize=100,w=1)
        db = client[self.db_name]

        train_collection = db[self.train_collection]
        test_collection = db[self.test_collection]
        valid_collection = db[self.valid_collection]

        datasets = {
            "train": MongoDBDataset(train_collection, self.batch_size),
            "valid": MongoDBDataset(test_collection, self.batch_size),
            "test": MongoDBDataset(valid_collection, self.batch_size)
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
        shapes = calc_shapes(self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullModel(shapes[0][1], shapes[1][2]).to(device)

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
            for batch_x1, batch_x2, batch_labels in tqdm(dataloader):
                batch_x1, batch_x2, batch_labels = batch_x1.to(device), batch_x2.to(device), batch_labels.to(device)

                if train:
                    optimizer.zero_grad()
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)
                if train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                correct += self.calculate_accuracy(outputs, batch_labels)
                total += batch_labels.size(0)

                all_preds.append(outputs.argmax(dim=1))
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
        self.model = FullModel(shapes[0][1], shapes[1][2]).to(device)
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
