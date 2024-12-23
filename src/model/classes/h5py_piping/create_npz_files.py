from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict
from chess_engine.src.model.classes.sqlite.database import  get_db
from chess_engine.src.model.config.config import data_settings
import os
import h5py

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


def db_to_hdf5_files():
    batch_file_size = data_settings.BatchFileSize  # Number of examples per file
    
    sets = {
        data_settings.TrainingDirectory: GamePositionRollup.is_training_data.is_(True),
        data_settings.TestingDirectory: GamePositionRollup.is_testing_data.is_(True),
        data_settings.ValidationDirectory: GamePositionRollup.is_validation_data.is_(True)
    }

    for h5_dir, filter_conditions in sets.items():
        
        delete_all_files(h5_dir)
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

                # Save to an HDF5 file
                batch_file_name = os.path.join(h5_dir, f'data_{idx}.h5')
                with h5py.File(batch_file_name, 'w') as f:
                    # Create datasets with compression
                    f.create_dataset('features', data=features_array, compression='gzip')
                    f.create_dataset('labels', data=labels_array, compression='gzip')

                print(f"Saved batch {idx} to {batch_file_name}")

                idx += 1


if __name__ == "__main__":
    db_to_hdf5_files()
