import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.config.config import Settings
import random
from joblib import dump, load, Parallel, delayed
import shutil
import csv
import math
import re
import os
import time

class data_generator():

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
 
    

    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

    def test_dataset_size(self,filename = None):
        if filename is None:
            filename = self.test_file
        return self.get_row_count(filename=filename)


    def copy_csv(self,source_file, destination_file):
        shutil.copy(source_file, destination_file)

    def preprocess_and_scale_chunk(self, chunk):
        # Preprocess the chunk. Adjust this to your actual preprocessing needs
        X, _, _ = self.clean_data(chunk)
        # Assuming X is now a clean, preprocessed DataFrame
        return X

    def create_scaler_cpu(self):

        
        scaler = StandardScaler()
        chunks = pd.read_csv(self.train_file, chunksize=100)
        
        # Preprocess chunks in parallel
        preprocessed_chunks = Parallel(n_jobs=-1)(
            delayed(self.preprocess_and_scale_chunk)(chunk) for chunk in chunks
        )
        
        # Combine all preprocessed chunks into one DataFrame if possible or fit scaler incrementally
        for preprocessed_chunk in preprocessed_chunks:
            scaler.partial_fit(preprocessed_chunk)
        
        self.init_scaler(scaler)

        return scaler


    def initialize_datasets(self):
        self.split_csv()
        self.headers = pd.read_csv(self.train_file,nrows=0)
        self.non_matrix_headers = [col for col in self.headers.columns if not col.endswith('positions')]
        self.create_scaler()
        #self.create_scaler_cpu()
        self.shape = self.get_shape()

    def split_csv(self, chunksize=10000):
        if os.path.exists(self.train_file):
            os.remove(self.train_file)
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.validation_file):
            os.remove(self.validation_file)
        if os.path.exists(self.copy_data):
            os.remove(self.copy_data)
        self.copy_csv(source_file=self.filename, destination_file=self.copy_data)
        
        self.filename = self.copy_data
        total_rows = self.get_row_count(filename=self.filename)

        #make sure no shared inices
        # Split indices for training+testing and validation
        validation_size = self.validation_size  # 20% of the data for validation
        train_test_indices = set(range(total_rows))
        validation_indices = set(random.sample(list(train_test_indices), int(total_rows * validation_size)))

        train_test_indices -= validation_indices  # Remove validation indices from training+testing pool

        # Further split training+testing indices into training and testing
        test_indices = set(random.sample(list(train_test_indices), int(len(train_test_indices) * self.test_size)))

        processed_rows = 0


        for chunk in pd.read_csv(self.filename, chunksize=chunksize):


            chunk_train = chunk.iloc[[i - processed_rows in train_test_indices and i - processed_rows not in test_indices for i in range(processed_rows, processed_rows + len(chunk))]]
            chunk_test = chunk.iloc[[i - processed_rows in test_indices for i in range(processed_rows, processed_rows + len(chunk))]]
            chunk_validation = chunk.iloc[[i - processed_rows in validation_indices for i in range(processed_rows, processed_rows + len(chunk))]]

            # Write to respective files
            mode = 'a' if processed_rows > 0 else 'w'
            chunk_train.to_csv(self.train_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_test.to_csv(self.test_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_validation.to_csv(self.validation_file, mode=mode, index=False, header=(mode == 'w'))

            # Update processed rows counter
            processed_rows += len(chunk)

    def clean_data(self, data):
        # Create DataFrame for columns with 'positions' suffix
        if data is None:
            return None,None,None
        matrix_data = pd.DataFrame()
        for col in self.matrix_headers:
            matrix_data[col] = data[col].apply(lambda x: flat_string_to_array(x))

        # DataFrame without 'positions' columns


        # Split the non-positions data into features and target
        X = data[self.non_matrix_headers]
        X = X.drop(columns=self.target_feature)
        Y = data[self.target_feature]

        return X, Y, matrix_data
    
    def create_scaler(self):
        scaler = StandardScaler()
        
        limit = self.get_row_count(self.train_file)
        adjusted_limit = math.ceil(limit/self.scalarBatchSize)
        batch_amt = 0
        batches = self.data_generator_no_matrices(batch_size=self.scalarBatchSize,filename=self.train_file)
        for batch in  tqdm(batches, total=adjusted_limit, desc="Creating Scalar"):
            if batch_amt > (limit-1):
                break
            else:

                scaler.partial_fit(batch[1])
                batch_amt += self.scalarBatchSize
        self.init_scaler(scaler=scaler)

        return scaler
    
    def data_generator(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                X, Y, matrix_data = self.clean_data(chunk)
                Y = np.array(Y)
                #output = np.reshape(Y,(-1,1))
                output = Y
                yield (matrix_data, X, output )

    def data_generator_no_matrices(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size,usecols=self.non_matrix_headers)
            for chunk in data:
                X, Y, matrix_data = self.clean_data(chunk)
                Y = np.array(Y)

                output = Y
                yield (matrix_data, X, output )
                
    def scaled_data_generator(self, batch_size,filename):


        while True:
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                    X, Y, matrixData = self.clean_data(chunk)
                    X_scaled = self.scaler.transform(X)
                    #print(matrixData.columns)
                    matrixData_scaled = matrixData.stack().map(reshape_to_matrix).unstack()

                    output_matrices = np.stack(matrixData_scaled.apply(lambda row: np.stack(row, axis=-1), axis=1).to_numpy())

                    Y = np.array(Y)

                    output = Y
                    yield ((output_matrices,X_scaled), output)


    def get_shape(self):
        self.headers = pd.read_csv(self.train_file,nrows=0)
        self.non_matrix_headers = [col for col in self.headers.columns if not col.endswith('positions')]
        self.matrix_headers = [col for col in self.headers.columns if col.endswith('positions')]
        matrix_channels = 0
        metadata_columns = 0
        batch = next(self.data_generator(batch_size=self.gen_batch_size,filename=self.train_file))
        matrix_channels = len(batch[0].columns)
        metadata_columns = len(batch[1].columns)
        matrix_shape = (8,8,matrix_channels)
        metadata_shape = (metadata_columns,)
        self.shape = [matrix_shape,metadata_shape]

        self.train_data = self.dataset_from_generator(filename=self.train_file)
        self.test_data = self.dataset_from_generator(filename=self.test_file)
        self.validation_data = self.dataset_from_generator(filename=self.validation_file)
        self.load_scaler()


        return self.shape
    
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
                    
def string_to_array(s):
    # Safely evaluate the string as a Python literal (list of lists in this case)
    return np.array(ast.literal_eval(s))



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