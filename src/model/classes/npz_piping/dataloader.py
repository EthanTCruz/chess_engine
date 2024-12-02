import os
import bisect
import torch
from torch.utils.data import Dataset
import numpy as np
from chess_engine.src.model.config.config import  dl_settings

# Define the NpzDataset class
class NpzDataset(Dataset):
    def __init__(self, data_directory, max_cache_size = dl_settings.MaxCacheSize , transform=None, target_transform=None):
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
        self.max_cache_size = max_cache_size # Adjust based on available memory
        
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