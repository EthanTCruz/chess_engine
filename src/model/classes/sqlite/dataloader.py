from torch.utils.data import Dataset, DataLoader
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import torch
from chess_engine.src.model.classes.bitboard_processing import sample_bitboard_dict,bc
from chess_engine.src.model.classes.sqlite.database import  get_db

class SQLAlchemyDataset(Dataset):
    def __init__(self, model_class, db_url, attributes_dict, batch_size):
        # Set up database connection
        self.engine = create_engine(db_url)
        self.Session = get_db()
        self.model_class = model_class
        self.attributes_dict = attributes_dict

        # Fetch the total number of records
        with next(self.Session) as session:
            self.length = session.query(model_class).count()


    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with self.Session() as session:
            # Fetch the specific record
            record = session.query(self.model_class).offset(idx).limit(1).first()
            
            # Extract attributes based on the provided dictionary
            features = [getattr(record, attr) for attr in self.attributes_dict.keys()]
            
            # Generate bitboards for the record
            bitboards = self.bitboard_creator.generate_bitboards(features)
            
            # Convert to tensors (example assumes integer bitboards)
            bitboard_tensor = torch.tensor(bitboards, dtype=torch.float32)
            
            # Set up labels if necessary (e.g., win_buckets or other attributes)
            label = torch.tensor(record.win_buckets, dtype=torch.float32)
            
            return bitboard_tensor, label