from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict
from chess_engine.src.model.classes.autoencoder.FeatureExtractor import get_metadata_from_gpr, sample_metadata
from chess_engine.src.model.classes.sqlite.database import get_db
from chess_engine.src.model.config.config import data_settings
import os
import h5py

def delete_all_files(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        return
    
    # Iterate over all files in the directory and delete them
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def db_to_hdf5_files(batch_retrieval_size: int = data_settings.BATCH_SIZE,
                      chunk_size: int = data_settings.H5PY_CHUNK_SIZE):
    """
    This version creates ONE h5 file for training, ONE for testing, ONE for validation,
    each containing chunked, preallocated datasets ('features', 'flattened_features',
    'metadata', and 'labels'). Preallocating the datasets avoids the overhead of repeated resizing.
    """
    num_bitboards = len(sample_bitboard_dict.keys())
    num_metada = len(sample_metadata.keys())
    flattened_num_bitboards = num_bitboards * 8 * 8 + num_metada

    # Define the splits and their corresponding filter conditions
    sets = {
        data_settings.TRAINING_DIR: GamePositionRollup.is_training_data.is_(True),
        data_settings.TESTING_DIR: GamePositionRollup.is_testing_data.is_(True),
        data_settings.VALIDATION_DIR: GamePositionRollup.is_validation_data.is_(True),
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
            # Preallocate datasets with total_records rows
            features_dset = h5f.create_dataset(
                "features",
                shape=(total_records, num_bitboards, 8, 8),
                dtype="uint64",
                chunks=(chunk_size, num_bitboards, 8, 8),
                compression="gzip"
            )
            flattened_features_dset = h5f.create_dataset(
                "flattened_features",
                shape=(total_records, flattened_num_bitboards),
                dtype="uint64",
                chunks=(chunk_size, flattened_num_bitboards),
                compression="gzip"
            )
            metadata_dset = h5f.create_dataset(
                "metadata",
                shape=(total_records, num_metada),
                dtype="uint64",
                chunks=(chunk_size, num_metada),
                compression="gzip"
            )
            labels_dset = h5f.create_dataset(
                "labels",
                shape=(total_records, 3),
                dtype="float32",
                chunks=(chunk_size, 3),
                compression="gzip"
            )

            current_index = 0

            with next(get_db()) as session:
                with tqdm(total=total_records, desc=f"[{h5_dir}] Writing Records", unit=" records") as pbar:
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
                        flattened_features_list = []
                        metadata_list = []
                        labels_list = []

                        for record in records:
                            # Extract features and metadata
                            bitboard_values = [getattr(record, attr) for attr in sample_bitboard_dict.keys()]
                            features = bitboards_to_array(bitboard_values)
                            metadata = get_metadata_from_gpr(record)

                            # Concatenate flattened features and metadata
                            flattened_features = np.concatenate(
                                (features.reshape(1, -1), metadata.reshape(1, -1)),
                                axis=1
                            )

                            # Extract labels and adjust based on turn
                            labels = record.win_buckets
                            if record.turn == 'b':
                                labels = labels[::-1]
                                features = features[:, ::-1, ::-1]

                            features_list.append(features)
                            flattened_features_list.append(flattened_features)
                            metadata_list.append(metadata)
                            labels_list.append(labels)

                        # Convert lists to NumPy arrays
                        features_array = np.array(features_list, dtype=np.float32)
                        flattened_features_array = np.array(flattened_features_list, dtype=np.float32).squeeze(axis=1)
                        metadata_array = np.array(metadata_list, dtype=np.float32)
                        labels_array = np.array(labels_list, dtype=np.float32)

                        batch_size = features_array.shape[0]

                        # Write data directly into the preallocated slices
                        features_dset[current_index:current_index + batch_size, ...] = features_array
                        flattened_features_dset[current_index:current_index + batch_size, ...] = flattened_features_array
                        metadata_dset[current_index:current_index + batch_size, ...] = metadata_array
                        labels_dset[current_index:current_index + batch_size, ...] = labels_array

                        current_index += batch_size
                        pbar.update(batch_size)

if __name__ == "__main__":
    db_to_hdf5_files()
