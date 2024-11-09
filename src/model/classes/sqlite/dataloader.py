from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import torch
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict, Bitboard_Creator
from chess_engine.src.model.classes.sqlite.database import  get_db

class SQLAlchemyDataset(Dataset):
    def __init__(self, model_class,  batch_size):

        self.Session = get_db()
        self.model_class = model_class
        self.attributes_dict = sample_bitboard_dict
        self.bc = Bitboard_Creator()
        self.batch_size = batch_size

        # Fetch the total number of records
        with next(get_db()) as session:
            self.length = session.query(model_class).count()
            print(self.length)


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with next(get_db()) as session:
            # Fetch the specific record
            record = session.query(self.model_class).offset(idx).limit(self.batch_size).first()
            

            if record is None:
                print(f"No record found at index {idx}")
                return None
            
            # print(f"Fetched record at index {idx}: {record}")

            # Extract attributes based on the provided dictionary
            features = [getattr(record, attr) for attr in self.attributes_dict.keys()]
            
            # print("Extracted features:", features)

            # Generate bitboards for the record
            bitboards = bitboards_to_array(features)
            
            # Convert to tensors (example assumes integer bitboards)
            bitboard_tensor = torch.tensor(bitboards, dtype=torch.float32)
            
            # Set up labels if necessary (e.g., win_buckets or other attributes)
            label = torch.tensor(record.win_buckets, dtype=torch.float32)
            
            return bitboard_tensor, label