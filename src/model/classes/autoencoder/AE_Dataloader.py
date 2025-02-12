import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from chess_engine.src.model.config.config import data_settings, ae_settings
import h5py
import time


# Global dictionary {worker_id: h5_file_object}
_worker_h5_handles = {}

def worker_init_fn(worker_id, h5_path):
    global _worker_h5_handles
    if h5_path:
        _worker_h5_handles[worker_id] = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        # print(f"Worker {worker_id} initialized successfully.")

class FlattenTransform:
    def __call__(self, features):
        """
        Flattens the feature tensor into a 1D vector.
        """
        return torch.from_numpy(features).float().view(-1)  # Shape: [832]



class HDF5SingleFileDataset(Dataset):
    """
    A Dataset that reads from one chunked HDF5 file with datasets:
      - "features" of shape (N, num_bitboards, 8, 8)
      - "labels" of shape (N, 3)
    """
    def __init__(self, h5_path, transform=None,device=torch.device('cpu')):
        """
        Args:
            h5_file_path (str): Path to the .h5 file ('data_all.h5').
            transform (callable, optional): A transform to apply to the features.
        """
        super().__init__()
        self.h5_file_path = f"{h5_path}/data_all.h5"
        assert os.path.exists(self.h5_file_path), f"HDF5 file not found at {self.h5_file_path}"
        self.transform = transform
        self.device = device
        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as h5f:
            self.length = h5f['features'].shape[0]



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the worker ID
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            with h5py.File(self.h5_file_path, 'r') as hf:
                if self.transform:
                    flattened_features = hf["flattened_features"][idx]
                    labels = hf["labels"][idx]
                else:
                    features = hf["features"][idx]
                    metadata = hf["metadata"][idx]
                    labels = hf["labels"][idx]

        else:
            worker_id = worker_info.id
            if worker_id not in _worker_h5_handles:
                raise KeyError(f"Worker {worker_id} does not have an HDF5 handle. Available: {_worker_h5_handles.keys()}")
            hf = _worker_h5_handles[worker_id]
            if self.transform:
                flattened_features = hf["flattened_features"][idx]
                labels = hf["labels"][idx]
            else:
                features = hf["features"][idx]
                metadata = hf["metadata"][idx]
                labels = hf["labels"][idx]

        labels_tensor = torch.from_numpy(labels).float().to(self.device)
        # Apply any transform you want to the features
        if self.transform:
            flattened_features = torch.from_numpy(flattened_features).float().to(self.device)

            return flattened_features, labels_tensor


        metadata_tensor = torch.from_numpy(metadata).float().to(self.device)
        features_tensor = torch.from_numpy(features).float().to(self.device)
        
        return features_tensor, metadata_tensor, labels_tensor

class InMemoryHDF5Dataset(Dataset):
    """
    A Dataset that loads the entire HDF5 file into memory.
    Assumes that the file contains the datasets 'features' and 'labels'
    (and optionally 'metadata').
    """
    def __init__(self, h5_path, transform=None, device=torch.device('cpu')):
        """
        Args:
            h5_path (str): Directory containing data_all.h5.
            transform (callable, optional): A transform to apply to the features.
            device (torch.device, optional): The device to move the tensors to.
        """
        self.h5_file_path = os.path.join(h5_path, "data_all.h5")
        assert os.path.exists(self.h5_file_path), f"HDF5 file not found at {self.h5_file_path}"
        self.transform = transform
        self.device = device
        
        # Open the file and load all data into memory
        with h5py.File(self.h5_file_path, 'r') as h5f:
            if self.transform:
                # If using transform, we assume you have precomputed flattened features
                self.features = h5f["flattened_features"][:]  # e.g., shape (N, D)
            else:
                self.features = h5f["features"][:]             # e.g., shape (N, num_bitboards, 8, 8)
                self.metadata = h5f["metadata"][:]             # e.g., shape (N, ?)
            self.labels = h5f["labels"][:] 
        
        self.length = self.labels.shape[0]
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Convert the data to tensors and move to the desired device
        if self.transform:
            features_tensor = torch.from_numpy(self.features[idx]).float().to(self.device)
            labels_tensor = torch.from_numpy(self.labels[idx]).float().to(self.device)
            return features_tensor, labels_tensor
        else:
            features_tensor = torch.from_numpy(self.features[idx]).float().to(self.device)
            metadata_tensor = torch.from_numpy(self.metadata[idx]).float().to(self.device)
            labels_tensor = torch.from_numpy(self.labels[idx]).float().to(self.device)
            return features_tensor, metadata_tensor, labels_tensor

def get_inmemory_dataloader(h5_path, batch_size, shuffle, num_workers, transform=None):
    dataset = InMemoryHDF5Dataset(h5_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,  # You can still use workers for collating/processing the batches.
        pin_memory=True  # Optionally, pin memory for faster GPU transfer.
    )
    return loader


def get_dataloader(h5_path, 
                    batch_size=ae_settings.DATALOADER_BATCH_SIZE, 
                    shuffle=True, 
                    num_workers=ae_settings.NUM_WORKERS,
                    transform=None,
                    prefetch_factor=None,
                    persistent_workers=False):
    
    dataset = HDF5SingleFileDataset(h5_path,transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, dataset.h5_file_path),
        shuffle=shuffle,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=ae_settings.USE_PIN_MEMORY
    )
    return loader

def get_dataloaders(transform,
                    batch_size=ae_settings.DATALOADER_BATCH_SIZE,
                    num_workers=ae_settings.NUM_WORKERS,
                    prefetch_factor=None,
                    persistent_workers=ae_settings.PERSIST_WORKERS):
    if num_workers > 0:
        prefetch_factor = 2

    train_loader = get_dataloader(data_settings.TRAINING_DIR,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              transform=transform,
                              prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
    test_loader = get_dataloader(data_settings.TESTING_DIR,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  transform=transform,
                                  prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
    valid_loader = get_dataloader(data_settings.VALIDATION_DIR,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  transform=transform,
                                  prefetch_factor=prefetch_factor,
                              persistent_workers=persistent_workers)
    return train_loader, test_loader, valid_loader



def get_dataloader_full_retrieval_time():
    num_epochs = 1
    train_loader = get_dataloader(data_settings.TRAINING_DIR, 
                                  batch_size=ae_settings.DATALOADER_BATCH_SIZE, 
                                  shuffle=True, 
                                  num_workers=ae_settings.NUM_WORKERS)
    start = time.time()
    i = 0
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            i = i + features.shape[0]
            # print(f"feature shape: {features.shape}, labels shape: {labels.shape}")
            pass
            # features => shape (64, 12, 8, 8)
            # labels   => shape (64, 3)
            # your training logic here...
    end = time.time()
    elapsed_time = end - start
    print(f"total run time: {elapsed_time}, training examples: {i}")
    return elapsed_time
