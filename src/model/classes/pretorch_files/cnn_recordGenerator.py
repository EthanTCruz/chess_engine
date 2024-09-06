import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.pretorch_files.cnn_game_analyzer import game_analyzer

import random
from joblib import dump, load, Parallel, delayed
import shutil
import csv
import math
import re
import os
import time

metadata_key = 'metadata'
bitboards_key = 'positions_data'
results_key = 'game_results'
feature_description = {
    'bitboards': tf.io.FixedLenFeature([], tf.string),
    'metadata': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.string),
}

class record_generator():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        


        if "epochs" not in kwargs:
            self.epochs=100
        else:
            self.epochs = kwargs["epochs"]

        if "target_feature" not in kwargs:
            self.target_feature = "w/b"
        else:
            self.target_feature = kwargs["target_feature"]
        
        if "predictions_board" not in kwargs:
            self.predictions_board='predictions.csv'
        else:
            self.predictions_board = kwargs["predictions_board"]

        if "batch_size" not in kwargs:
            self.batch_size = s.nnBatchSize
        else:
            self.batch_size = kwargs["batch_size"]

        
                
        if "scalerFile" not in kwargs:
            self.scalerFile = s.scaler_weights
        else:
            self.scalerFile = kwargs["scalerFile"]

        if "matrixScalerFile" not in kwargs:
            self.matrixScalerFile = s.matrixScalerFile
        else:
            self.matrixScalerFile = kwargs["matrixScalerFile"]

        if "filename" not in kwargs:
            self.filename = s.scores_file
        else:
            self.filename = kwargs["filename"]


        if "test_size" not in kwargs:
            self.test_size=.2
        else:
            self.test_size=kwargs["test_size"]

        
        if "random_state" not in kwargs:
            self.random_state=42
        else:
            self.random_state = kwargs["random_state"]

        self.train_file = s.trainingFile
        self.test_file = s.testingFile
        self.gen_batch_size = s.nnGenBatchSize
        self.copy_data = s.copy_file
        self.validation_file = s.validationFile
        self.validation_size = s.nnValidationSize
        self.scalarBatchSize = s.nnScalarBatchSize

        self.recordsDataFile = s.recordsData

        self.recordsDataFileCopy = s.recordsDataCopy
        self.recordsDataFileTrain = s.recordsDataTrain
        self.recordsDataFileTest = s.recordsDataTest
        self.recordsDataFileValidation = s.recordsDataValidation

        evaluator = game_analyzer(scores_file=s.scores_file)
        evaluator.set_feature_description()
        self.feature_description = evaluator.get_feature_description()

        self.seed=3141


    
    def init_scaler(self,scaler:StandardScaler,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        dump(scaler, scalarFile)
        self.scalar = scaler
        return 0
    

    def load_scaler(self,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        self.scaler = load(scalarFile)
        return self.scaler
 
    
    def _parse_function(self,example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        bitboards = tf.io.parse_tensor(example['bitboards'], out_type=tf.int8)
        metadata = tf.io.parse_tensor(example['metadata'], out_type=tf.float16)
        target = tf.io.parse_tensor(example['target'], out_type=tf.float16)
        return bitboards, metadata, target
    
    
    def parser(self,recordsFile):

        dataset = tf.data.TFRecordDataset([recordsFile])
        parsed_dataset = dataset.map(self._parse_function)

        return parsed_dataset


    def metadata_generator_records(self):
        for _, metadata, _ in self.parsed_dataset:
            yield metadata
    
    def get_row_count_records(self):
        # Initialize a counter
        count = 0
        for _ in self.parsed_dataset:
            count += 1
        return count

    def get_row_count_records_old(self):
        # Assuming the dataset is batched
        return sum(1 for _ in self.parsed_dataset.unbatch())

    def create_scaler_records(self):
        scaler = StandardScaler()
        
        # Adjusted for TensorFlow's dataset API
        total_rows = count_dataset_elements(dataset=self.parsed_dataset)
        adjusted_limit = math.ceil(total_rows / self.scalarBatchSize)
        
        # Convert dataset to generator for partial fitting
        batches = self.metadata_generator_records()
        
        # Loop through the dataset and partial_fit the scaler
        for _ in range(adjusted_limit):
            try:
                batch = next(batches)
                batch = np.array(batch).reshape(1, -1) 
                scaler.partial_fit(batch)
            except StopIteration:
                break  # When dataset ends

        self.init_scaler(scaler=scaler)
        return scaler

    def initialize_datasets_records(self):
        self._split_tfrecord()
        self.parsed_dataset = self.parser(recordsFile=self.recordsDataFileTrain)
        self.create_scaler_records()
        # #self.create_scaler_cpu()
        self.shape = self.get_shape()

    def dataset_from_generator(self,filename,batch_size: int = None):
        if batch_size is None:
            batch_size = self.gen_batch_size
        shapes = self.shape
        matrix_shape = shapes[0]
        meta_shape = shapes[1]
        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_data_generator(filename=filename, batch_size=batch_size),
            output_types=((tf.int8, tf.float16),tf.float16),  
            output_shapes=(([self.gen_batch_size, matrix_shape[0], matrix_shape[1], matrix_shape[2]], 
                            [self.gen_batch_size, meta_shape[0]]), 
                            [self.gen_batch_size, 3]) 
        )
        return dataset

    def set_shape_record(self):
        # [matrix_shape,metadata_shape]
        row = 8
        col = 8
        for bitboards, metadata, target in self.parsed_dataset.take(1):
            self.input_shape = [(8,8,bitboards.shape[0]),(metadata.shape[0],)]
            self.shape = (([self.gen_batch_size,row,col,bitboards.shape[0]],
                           [self.gen_batch_size,metadata.shape[0]]),
                          [self.gen_batch_size,target.shape[0]])
        return self.shape

    def scale_features(self,data):

        features = np.array(data).reshape(1, -1)  # Reshape to 2D array
        
        # Scale the features using the loaded scaler
        scaled_features = self.scaler.transform(features)
        
        # # If you need to convert back to TensorFlow tensors (optional)
        # # Example: converting the entire scaled array back into a dictionary of tensors with the same keys as the original
        # scaled_other = {key: tf.convert_to_tensor(value, dtype=tf.float32) 
        #                 for key, value in zip(data.keys(), scaled_features.flatten())}

        return scaled_features

    def test_dataset_size(self):
        return count_dataset_elements()

    def record_generator(self,recordsFile,batch_size: int = 5):
        tfrecord_filenames = [recordsFile]
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        
        parsed_dataset = dataset.map(self._parse_function)

        for record in parsed_dataset:
            # scaled_metadata = self.scale_features(record[1])

            yield ((record[0],record[1]), record[2])

    def scaled_record_generator(self,recordsFile,batch_size: int = 5):
        tfrecord_filenames = [recordsFile]
        dataset = tf.data.TFRecordDataset(tfrecord_filenames)
        
        parsed_dataset = dataset.map(self._parse_function)

        for record in parsed_dataset.take(batch_size):
            bitboards = tf.reshape(record[0], [1,8,8,self.shape[0][0][3]])
            scaled_metadata = self.scale_features(record[1])
            target = tf.reshape(record[2],[1,3])
            yield ((bitboards,scaled_metadata), target)

    def scaled_dataset_generator(self,recordsFile: str,batch_size: int = 10):

        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_record_generator(recordsFile=recordsFile,batch_size=batch_size),
            output_types=((tf.float16, tf.float16), tf.float16),
            output_shapes=self.shape
        )
        return dataset

    def get_shape(self):
            self.parsed_dataset = self.parser(recordsFile=self.recordsDataFileTrain)
            self.set_shape_record()

            self.load_scaler()
            self.train_data = self.scaled_dataset_generator(recordsFile=self.recordsDataFileTrain,batch_size=self.gen_batch_size)
            self.test_data = self.scaled_dataset_generator(recordsFile=self.recordsDataFileTest,batch_size=self.gen_batch_size)
            self.validation_data = self.scaled_dataset_generator(recordsFile=self.recordsDataFileValidation,batch_size=self.gen_batch_size)

            return self.input_shape
    def _split_tfrecord(self):
        # random.seed(3141)
        # Load the dataset

        if os.path.exists(self.recordsDataFileTrain):
            os.remove(self.recordsDataFileTrain)
        if os.path.exists(self.recordsDataFileTest):
            os.remove(self.recordsDataFileTest)
        if os.path.exists(self.recordsDataFileValidation):
            os.remove(self.recordsDataFileValidation)
        if os.path.exists(self.recordsDataFileCopy):
            os.remove(self.recordsDataFileCopy)

        copy_csv(self.recordsDataFile,self.recordsDataFileCopy)
        self.recordsDataFile = self.recordsDataFileCopy

        raw_dataset = tf.data.TFRecordDataset(self.recordsDataFile)

        # Define split ratios
        train_ratio = 1.0 - (self.test_size + self.validation_size)
        
        # Probabilistically filter the dataset into train, validation, and test sets
        def is_train(x):
            return tf.random.uniform([], seed=self.seed) < train_ratio

        def is_val(x):
            return tf.logical_and(tf.random.uniform([], seed=self.seed) >= train_ratio, 
                                  tf.random.uniform([], seed=self.seed) < train_ratio + self.validation_size)

        def is_test(x):
            return tf.random.uniform([], seed=self.seed) >= (train_ratio + self.validation_size)

        train_dataset = raw_dataset.filter(is_train)
        val_dataset = raw_dataset.filter(is_val)
        test_dataset = raw_dataset.filter(is_test)

        # Function to write the split datasets to new TFRecord files
        def write_tfrecord(dataset, filename):
            tf.data.experimental.TFRecordWriter(filename).write(dataset)
        
        # Write the splits to new TFRecord files
        write_tfrecord(train_dataset, self.recordsDataFileTrain)
        write_tfrecord(val_dataset, self.recordsDataFileValidation)
        write_tfrecord(test_dataset, self.recordsDataFileTest)
        



                        
def string_to_array(s):
    # Safely evaluate the string as a Python literal (list of lists in this case)
    return np.array(ast.literal_eval(s))

def count_dataset_elements(dataset):
    # Attempt to use the cardinality if available
    cardinality = tf.data.experimental.cardinality(dataset).numpy()
    if cardinality in [tf.data.experimental.INFINITE_CARDINALITY, tf.data.experimental.UNKNOWN_CARDINALITY]:
        # Fallback: Iterate through the dataset and count elements
        return sum(1 for _ in dataset)
    else:
        return cardinality

def flat_string_to_array(s):
    if not s:
        return None

    # Remove all square brackets and replace newline characters with spaces
    clean_string = s.replace('[', '').replace(']', '').replace('\n', ' ')

    # Split the cleaned string on spaces to isolate the numbers as strings
    numbers_str = clean_string.split()

    # Convert list of number strings to a NumPy array of type int8
    try:
        integer_array = np.array(numbers_str, dtype=np.int8)
    except ValueError as e:
        # Handle cases where conversion fails due to invalid numeric strings
        print(f"Error converting string to array: {e}")
        return None

    return integer_array



def reshape_to_matrix(cell):
    return np.array(cell).reshape(8, 8)

def copy_csv(source_file, destination_file):
    shutil.copy(source_file, destination_file)