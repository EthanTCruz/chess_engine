import torch
from torch.utils.data import Dataset, DataLoader
from chess_engine.src.model.classes.mongo_functions import mongo_data_pipe,iteratingFunctionScaled
from pymongo import MongoClient
from chess_engine.src.model.config.config import Settings
import numpy as np

class MongoDBDataset(Dataset):
    def __init__(self, collection, batch_size=1024):
        s = Settings()

        self.means = np.load(s.np_means_file)
        self.stds = np.load(s.np_stds_file)
        
        self.batch_size = batch_size
        


        self.collection = collection


        
        self.data = []

        # Use a process-safe way to load data
        self._load_data()
        
        # Fetch data with a progress bar

    def _load_data(self):
        try:
            for doc in iteratingFunctionScaled(collection=self.collection,
                                               batch_size=self.batch_size,
                                               means = self.means,
                                               stds = self.stds):
                self.data.append(doc)
            print(f"Loaded {len(self.data)} documents.")
        except Exception as e:
            print("Error during data loading:", e)
            

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            if idx >= len(self.data):
                raise IndexError("Index out of range")
            
            doc = self.data[idx]
            positions_data = doc['positions_data']
            metadata = doc['metadata']
            game_results = doc['game_results']
            
            # print(f"Returning item {idx} from dataset")
            
            return (torch.tensor(positions_data, dtype=torch.float32),
                    torch.tensor(metadata, dtype=torch.float32),
                    torch.tensor(game_results, dtype=torch.float32))
        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            raise

        
