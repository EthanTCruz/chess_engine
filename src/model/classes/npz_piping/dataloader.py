from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict, Bitboard_Creator
from chess_engine.src.model.classes.sqlite.database import  get_db
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os






class NPZDataset(Dataset):
    def __init__(self, directory):
        self.data = []
        self.labels = []
        
        all_entries = os.listdir(directory)
        num_files = len([entry for entry in all_entries if os.path.isfile(os.path.join(directory, entry))])
        self.npz_files = [f'{directory}/data_{i}.npz' for i in range(num_files)]

        # Load all files into memory (optional)
        for file in self.npz_files:
            print(file)
            data = np.load(file)
            self.data.append(data['features'])
            self.labels.append(data['labels'])

        # Concatenate all data
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)





