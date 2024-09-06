import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from chess_engine.src.model.classes.MongoDBDataset import MongoDBDataset
from chess_engine.src.model.config.config import Settings
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from chess_engine.src.model.classes.cnn_bb_scorer import  calc_shapes

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
    

class model_operator():
    def __init__(self):

        s = Settings()
        
        self.trainCollectionName=s.training_collection_key
        self.testCollectionName=s.testing_collection_key
        self.validCollectionName=s.validation_collection_key

        self.mongoUrl=s.mongo_url
        self.dbName=s.db_name
        self.batch_size=s.BatchSize

        self.num_workers = s.num_workers
        self.model_path = s.torch_model_file

    def create_dataloaders(self,num_workers: int = 0):
        train_dataset = MongoDBDataset(collectionName=self.trainCollectionName, 
                        mongoUrl=self.mongoUrl, 
                        dbName=self.dbName, 
                        batch_size=self.batch_size)
        valid_dataset = MongoDBDataset(collectionName=self.validCollectionName, 
                    mongoUrl=self.mongoUrl, 
                    dbName=self.dbName, 
                    batch_size=self.batch_size)
        test_dataset = MongoDBDataset(collectionName=self.testCollectionName, 
                mongoUrl=self.mongoUrl, 
                dbName=self.dbName, 
                batch_size=self.batch_size)
    
        if len(train_dataset) == 0 or len(valid_dataset) == 0 or len(test_dataset) == 0:
            print("Dataset is empty. Please check the data loading process.")
            return 0
        else:

            train_dataloader = DataLoader(train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=True, 
                                    num_workers=num_workers)
            test_dataloader = DataLoader(test_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
            valid_dataloader = DataLoader(valid_dataset, 
                        batch_size=self.batch_size, 
                        shuffle=True, 
                        num_workers=num_workers)
            return train_dataloader, test_dataloader, valid_dataloader
    

    def Create_and_Train_Model(self, 
                               learning_rate: float = 0.001, 
                               num_epochs: int = 16, 
                               num_workers: int = 0,
                               save_model: bool = True):
        if num_workers < self.num_workers:
            num_workers = self.num_workers
        
        train_dataloader, test_dataloader, valid_dataloader = self.create_dataloaders(num_workers=num_workers)
        shapes = calc_shapes(batch_size=self.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullModel(shapes[0][1], shapes[1][2]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0
            correct_samples = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_x1, batch_x2, batch_labels in progress_bar:
                batch_x1, batch_x2, batch_labels = batch_x1.to(device), batch_x2.to(device), batch_labels.to(device)
                # b1 = [1024,13,8,8]
                optimizer.zero_grad()
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total_samples += batch_x1.size(0)
                correct_samples += self.calculate_accuracy(outputs, batch_labels)
            
            avg_train_loss = running_loss / len(train_dataloader)
            train_accuracy = correct_samples / total_samples * 100
            
            avg_val_loss, val_accuracy, val_predictions, val_labels = self.evaluate(model, valid_dataloader, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        test_loss, test_accuracy, test_predictions, test_labels = self.evaluate(model, test_dataloader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        if save_model:
            self.save_model(model=model, optimizer=optimizer, model_path=self.model_path)
        # Generate confusion matrix
        cm = confusion_matrix(test_labels.cpu(), test_predictions.cpu())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        

        return model
    
    def evaluate_with_indices(self, model, data_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        total_samples = 0
        correct_samples = 0
        all_predictions = []
        all_labels = []
        incorrect_indices = []
        
        with torch.no_grad():
            for batch_idx, (batch_x1, batch_x2, batch_labels) in enumerate(data_loader):
                batch_x1, batch_x2, batch_labels = batch_x1.to(device), batch_x2.to(device), batch_labels.to(device)
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)
                running_loss += loss.item()
                total_samples += batch_x1.size(0)
                correct_samples += self.calculate_accuracy(outputs, batch_labels)
                
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(batch_labels, 1)
                all_predictions.append(predicted)
                all_labels.append(labels)
                
                # Store indices of incorrect predictions
                incorrect_indices.extend([batch_idx * data_loader.batch_size + i for i in range(batch_labels.size(0)) if predicted[i] != labels[i]])
        
        avg_loss = running_loss / len(data_loader)
        accuracy = correct_samples / total_samples * 100
        
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)

        return avg_loss, accuracy, all_predictions, all_labels, incorrect_indices


    def load_and_evaluate_model(self, model_path, num_workers: int = 0):
        if num_workers < self.num_workers:
            num_workers = self.num_workers

        _, test_dataloader, _ = self.create_dataloaders(num_workers=num_workers)
        shapes =  calc_shapes(batch_size=self.batch_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FullModel(shapes[0][1], shapes[1][2]).to(device)
        optimizer = optim.Adam(model.parameters())

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy, test_predictions, test_labels = self.evaluate(model, test_dataloader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Generate confusion matrix
        cm = confusion_matrix(test_labels.cpu(), test_predictions.cpu())
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

            # Calculate precision, recall, and F1 score
        precision = precision_score(test_labels.cpu(), test_predictions.cpu(), average=None)
        recall = recall_score(test_labels.cpu(), test_predictions.cpu(), average=None)
        f1 = f1_score(test_labels.cpu(), test_predictions.cpu(), average=None)

        for i in range(len(precision)):
            print(f"Class {i}:")
            print(f"  Precision: {precision[i]:.4f}")
            print(f"  Recall: {recall[i]:.4f}")
            print(f"  F1-Score: {f1[i]:.4f}")

    def save_model(self, model, optimizer, model_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")


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

    def predict_single_example(self,bitboards, metadata):
        """
        Uses the loaded model to make a prediction on a single example.

        Arguments:
        model -- The loaded PyTorch model.
        bitboard_example -- The bitboard input, a single example as a tensor of shape (1, n_C, H, W).
        metadata_example -- The metadata input, a single example as a tensor of shape (1, n_features).

        Returns:
        prediction -- The model's predicted class for the input example.
        """
        # self.model.eval()  # Set the model to evaluation mode

        # Ensure the example tensors are on the same device as the model
        device = next(self.model.parameters()).device
        bitboards = bitboards.to(device)
        metadata = metadata.to(device)

        with torch.no_grad():  # No need to compute gradients
            output = self.model(bitboards, metadata)
            prediction = torch.argmax(output, dim=1)  # Get the index of the class with the highest probability


        return prediction.item()
        
    
    def evaluate(self, model, data_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        total_samples = 0
        correct_samples = 0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch_x1, batch_x2, batch_labels in data_loader:
                batch_x1, batch_x2, batch_labels = batch_x1.to(device), batch_x2.to(device), batch_labels.to(device)
                outputs = model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)
                running_loss += loss.item()
                total_samples += batch_x1.size(0)
                correct_samples += self.calculate_accuracy(outputs, batch_labels)
                
                _, predicted = torch.max(outputs, 1)
                _, labels = torch.max(batch_labels, 1)
                all_predictions.append(predicted)
                all_labels.append(labels)
        
        avg_loss = running_loss / len(data_loader)
        accuracy = correct_samples / total_samples * 100
        
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        return avg_loss, accuracy, all_predictions, all_labels

    def calculate_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs, 1)
        _, labels = torch.max(labels, 1)
        correct = (predicted == labels).sum().item()
        return correct
    
    

    # def calc_shapes(self,dataloader):

    #     for batch in dataloader:
    #         batch_x1, batch_x2, batch_labels = batch

    #         return batch_x1.shape, batch_x2.shape, batch_labels.shape

