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

        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file

        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



def db_to_hdf5_files(batch_retrieval_size: int = data_settings.BatchSize,
                                chunk_size: int = data_settings.ChunkSize):
    """
    This version creates ONE h5 file for training, ONE for testing, ONE for validation,
    each containing chunked, resizable datasets ('features' and 'labels'),
    with a tqdm progress bar for each data split.
    """

    num_bitboards = len(sample_bitboard_dict.keys())

    sets = {
        data_settings.TrainingDirectory: GamePositionRollup.is_training_data.is_(True),
        data_settings.TestingDirectory: GamePositionRollup.is_testing_data.is_(True),
        data_settings.ValidationDirectory: GamePositionRollup.is_validation_data.is_(True),
    }

    for h5_dir, filter_conditions in sets.items():

        delete_all_files(h5_dir)
        single_file_path = os.path.join(h5_dir, "data_all.h5")

        # Count total records in this split
        with next(get_db()) as session:
            total_records = session.query(GamePositionRollup)\
                                   .filter(filter_conditions)\
                                   .count()


        if total_records == 0:
            print(f"No records found for {h5_dir}, skipping.")
            continue

        with h5py.File(single_file_path, 'w') as h5f:
            # Create resizable, chunked datasets
            features_dset = h5f.create_dataset(
                "features",
                shape=(0, num_bitboards, 8, 8),
                maxshape=(None, num_bitboards, 8, 8),
                dtype="uint64",
                chunks=(chunk_size, num_bitboards, 8, 8),
                compression="gzip"
            )
            labels_dset = h5f.create_dataset(
                "labels",
                shape=(0, 3),
                maxshape=(None, 3),
                dtype="float32",
                chunks=(chunk_size, 3),
                compression="gzip"
            )

            current_size = 0

            with next(get_db()) as session:
                # Wrap the main loop with tqdm
                with tqdm(
                    total=total_records, 
                    desc=f"[{h5_dir}] Writing Records", 
                    unit=" records"
                ) as pbar:
                    for batch_start in range(0, total_records, batch_retrieval_size):

                        records = (session.query(GamePositionRollup)
                                  .filter(filter_conditions)
                                  .offset(batch_start)
                                  .limit(batch_retrieval_size)
                                  .all())

                        if not records:
                            break

                        # Collect batch data
                        features_list = []
                        labels_list = []

                        for record in records:
                            # Extract features
                            bitboard_values = [getattr(record, attr) 
                                               for attr in sample_bitboard_dict.keys()]
                            features = bitboards_to_array(bitboard_values)

                            # Extract labels
                            labels = record.win_buckets

                            if record.turn == 'b':
                                labels = labels[::-1]
                                features = features[:, ::-1, ::-1]

                            features_list.append(features)
                            labels_list.append(labels)

                        # Convert to NumPy arrays
                        features_array = np.array(features_list, dtype=np.float32)
                        labels_array   = np.array(labels_list, dtype=np.float32)

                        batch_size = features_array.shape[0]
                        new_size = current_size + batch_size

                        features_dset.resize((new_size, num_bitboards, 8, 8))
                        labels_dset.resize((new_size, 3))

                        features_dset[current_size:new_size, ...] = features_array
                        labels_dset[current_size:new_size, ...]   = labels_array
                        

                        current_size = new_size

                        pbar.update(batch_size)





if __name__ == "__main__":
    db_to_hdf5_files()
