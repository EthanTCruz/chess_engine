import os
import torch
from torch.utils.data import Dataset
import numpy as np
from chess_engine.src.model.config.config import data_settings

# Define the NpzDataset class
class NpzDataset(Dataset):
    def __init__(self, data_directory, transform=None, target_transform=None):
        """
        Custom Dataset for loading data from multiple .npz files into memory.

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

        self.data = []  # Preload all data here
        self.labels = []  # Preload all labels here

        # Load all .npz files into memory
        for file_name in sorted(os.listdir(data_directory)):
            if file_name.endswith('.npz'):
                file_path = os.path.join(data_directory, file_name)
                with np.load(file_path) as data:
                    self.data.append(data['features'])
                    self.labels.append(data['labels'])

        # Concatenate all data and labels to simplify indexing
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

        # Total samples
        self.total_samples = self.data.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.total_samples}")

        feature_sample = self.data[idx]
        label_sample = self.labels[idx]

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
