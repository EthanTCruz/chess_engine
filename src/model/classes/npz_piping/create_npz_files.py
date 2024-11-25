from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict
from chess_engine.src.model.classes.sqlite.database import  get_db
from chess_engine.src.model.config.config import settings
import os

def delete_all_files(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



def db_to_npz_files():

    batch_file_size = settings.npzBatchFileSize  # Number of examples per file
    
    sets = {settings.npzTrainingDirectory:GamePositionRollup.is_training_data.is_(True),
            settings.npzTestingDirectory:GamePositionRollup.is_testing_data.is_(True),
            settings.npzValidationDirectory: GamePositionRollup.is_validation_data.is_(True)}

    
    for npz_dir, filter_conditions in sets.items():
        
        delete_all_files(npz_dir)
        idx = 0
        with next(get_db()) as session:
            # Count total number of records
            total_records = session.query(GamePositionRollup).filter(
                filter_conditions
            ).count()
            print(f"Total records: {total_records}")

            # Process records in chunks
            for batch_start in range(0, total_records, batch_file_size):
                features_list = []
                labels_list = []

                # Fetch batch of records
                records = session.query(GamePositionRollup).filter(
                    filter_conditions
                ).offset(batch_start).limit(batch_file_size).all()

                for record in records:
                    # Extract features
                    features = [getattr(record, attr) for attr in sample_bitboard_dict.keys()]
                    features = bitboards_to_array(features)  # Convert to array format

                    # Extract labels
                    labels = record.win_buckets

                    # Append to batch
                    features_list.append(features)
                    labels_list.append(labels)

                # Convert lists to NumPy arrays
                features_array = np.array(features_list)  # Shape: (batch_file_size, 12, 8, 8)
                labels_array = np.array(labels_list)      # Shape: (batch_file_size, 3)

                # Save to an .npz file
                batch_file_name = f'{npz_dir}/data_{idx}.npz'
                np.savez_compressed(batch_file_name, features=features_array, labels=labels_array)
                print(f"Saved batch {idx} to {batch_file_name}")

                idx += 1

