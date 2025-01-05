import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from chess_engine.src.model.config.config import data_settings
import h5py
import time

# Global dictionary {worker_id: h5_file_object}
_worker_h5_handles = {}

def worker_init_fn(worker_id, h5_path):
    global _worker_h5_handles
    print(f"Initializing worker {worker_id} with HDF5 file path: {h5_path}")
    if h5_path:
        _worker_h5_handles[worker_id] = h5py.File(h5_path, 'r', libver='latest', swmr=True)
        print(f"Worker {worker_id} initialized successfully.")


class HDF5SingleFileDataset(Dataset):
    """
    A Dataset that reads from one chunked HDF5 file with datasets:
      - "features" of shape (N, num_bitboards, 8, 8)
      - "labels" of shape (N, 3)
    """
    def __init__(self, h5_path, transform=None):
        """
        Args:
            h5_file_path (str): Path to the .h5 file ('data_all.h5').
            transform (callable, optional): A transform to apply to the features.
        """
        super().__init__()
        self.h5_file_path = f"{h5_path}/data_all.h5"
        assert os.path.exists(self.h5_file_path), f"HDF5 file not found at {self.h5_file_path}"
        self.transform = transform

        with h5py.File(self.h5_file_path, 'r', libver='latest', swmr=True) as h5f:
            self.length = h5f['features'].shape[0]



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the worker ID
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            with h5py.File(self.h5_file_path, 'r') as hf:
                features = hf["features"][idx]
                labels = hf["labels"][idx]
        else:
            worker_id = worker_info.id
            if worker_id not in _worker_h5_handles:
                raise KeyError(f"Worker {worker_id} does not have an HDF5 handle. Available: {_worker_h5_handles.keys()}")
            hf = _worker_h5_handles[worker_id]
            features = hf["features"][idx]
            labels = hf["labels"][idx]

        # Apply any transform you want to the features
        if self.transform:
            features = self.transform(features)  # for example, normalization, etc.

        # Convert to torch tensors
        features_tensor = torch.from_numpy(features).float()   # shape: (num_bitboards, 8, 8)
        labels_tensor   = torch.from_numpy(labels).float()     # shape: (3,)

        return features_tensor, labels_tensor

def get_dataloader(h5_path, batch_size=32, shuffle=True, num_workers=4):
    dataset = HDF5SingleFileDataset(h5_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, dataset.h5_file_path),
        shuffle=shuffle
    )
    return loader


def get_dataloader_full_retrieval_time():
    num_epochs = 1
    train_loader = get_dataloader(data_settings.TrainingDirectory, batch_size=64, shuffle=True, num_workers=4)
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
