import ast
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.config.config import Settings
import random
from joblib import dump, load
import shutil
import csv
import math



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
        

    def get_shape_cnn(self):
        return self.cnn_shape

    def get_shape(self):
        return self.shape
        
    def dataset_from_generator(self,filename,batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_data_generator(filename=filename, batch_size=batch_size),
            output_types=(tf.float32, tf.float32),  # Update these types based on your data
            output_shapes=([None, self.shape], [None, 1])  # Update shapes based on your data
        )
        return dataset

    def create_scaler(self):
        scaler = StandardScaler()
        limit = self.get_row_count(self.train_file)
        batch_amt = 0
        for batch in self.data_generator(batch_size=self.gen_batch_size,filename=self.train_file):
            if batch_amt > (limit-1):
                break
            else:
                scaler.partial_fit(batch[0])
                batch_amt += self.gen_batch_size
        self.init_scaler(scaler=scaler)
        return scaler
    
    def init_scaler(self,scaler:StandardScaler,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        dump(scaler, scalarFile)
        return 0
    
    
    def initialize_datasets(self):
        self.shape = (self.count_csv_headers(filename=self.filename) - 2)
        self.copy_csv(source_file=self.filename, destination_file=self.copy_data)
        self.split_csv()
        self.create_scaler()

    def count_csv_headers(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Read the first row, which contains the headers
            return len(headers)
    
    def data_generator(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                X, Y = self.clean_data(chunk)
                Y = np.array(Y)
                output = np.reshape(Y,(-1,1))
                yield (X, output)

    def load_scaler(self,scalarFile: str = None):
        if scalarFile is None:
            scalarFile = self.scalerFile
        scaler = load(scalarFile)
        return scaler
    
    def scaled_data_generator(self, batch_size,filename):
        scaler = self.load_scaler()
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                    X, Y = self.clean_data(chunk)
                    X_scaled = scaler.transform(X)
                    Y = np.array(Y)
                    output = np.reshape(Y,(-1,1))
                    yield (X_scaled, output)
        


    def clean_data(self,data):
        data = data.drop(columns=['moves(id)'])

        w_indices = data[data['w/b'] == 'w'].index
        b_indices = data[data['w/b'] == 'b'].index


        new_indices = np.concatenate([w_indices, b_indices])


        data = data.loc[new_indices]
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)


        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])
        Y = data[self.target_feature]

        return X,Y
    

    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

    def test_dataset_size(self,filename = None):
        if filename is None:
            filename = self.test_file
        return self.get_row_count(filename=filename)

    def count_values_in_column(self,chunksize:int  = 10000):
        column_name = 'w/b'

        w_count = 0
        b_count = 0
        for chunk in pd.read_csv(self.filename, chunksize=chunksize, usecols=[column_name]):
            value_counts = chunk[column_name].value_counts()
            # Determine rows for train and test set
            w_count += value_counts.get('w')
            b_count += value_counts.get('b')
        limit = min(w_count,b_count)
        return limit


    def copy_csv(self,source_file, destination_file):
        shutil.copy(source_file, destination_file)




    def split_csv(self, chunksize=10000):
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
        processed_w_count = 0
        processed_b_count = 0
        max_wins = self.count_values_in_column(chunksize=chunksize)

        for chunk in pd.read_csv(self.filename, chunksize=chunksize):
            w_rows = chunk[chunk['w/b']=='w']
            b_rows = chunk[chunk['w/b']=='b']

            # Limit the rows based on processed count and min_count
            w_rows = w_rows.head(min(max_wins - processed_w_count, len(w_rows)))
            b_rows = b_rows.head(min(max_wins - processed_b_count, len(b_rows)))



            if processed_w_count >= max_wins:

                chunk_balanced = b_rows
            elif processed_w_count >= max_wins:

                chunk_balanced = w_rows
            else:
                chunk_balanced = pd.concat([w_rows, b_rows]).sample(frac=1)

            # Update processed counts
            processed_w_count += len(w_rows)
            processed_b_count += len(b_rows)


            chunk_train = chunk_balanced.iloc[[i - processed_rows in train_test_indices and i - processed_rows not in test_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]
            chunk_test = chunk_balanced.iloc[[i - processed_rows in test_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]
            chunk_validation = chunk_balanced.iloc[[i - processed_rows in validation_indices for i in range(processed_rows, processed_rows + len(chunk_balanced))]]

            # Write to respective files
            mode = 'a' if processed_rows > 0 else 'w'
            chunk_train.to_csv(self.train_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_test.to_csv(self.test_file, mode=mode, index=False, header=(mode == 'w'))
            chunk_validation.to_csv(self.validation_file, mode=mode, index=False, header=(mode == 'w'))

            # Update processed rows counter
            processed_rows += len(chunk)


    def initialize_cnn_datasets(self):
        self.copy_csv(source_file=self.filename, destination_file=self.copy_data)
        self.split_csv_cnn()
        self.create_scaler_cnn()
        self.cnn_shape = self.get_cnn_shape()

    def split_csv_cnn(self, chunksize=10000):
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

    def clean_data_cnn(self, data):
        # Create DataFrame for columns with 'positions' suffix
        matrix_data = pd.DataFrame()
        for col in data.columns:
            if col.endswith('positions'):
                matrix_data[col] = data[col].apply(lambda x: flat_string_to_array(x))

        # DataFrame without 'positions' columns
        non_positions_data = data.drop(columns=matrix_data.columns)

        # Split the non-positions data into features and target
        X = non_positions_data.drop(columns=self.target_feature)
        Y = non_positions_data[self.target_feature]

        return X, Y, matrix_data
    
    def create_scaler_cnn(self):
        scaler = StandardScaler()
        #matrixScaler = StandardScaler()
        
        limit = self.get_row_count(self.train_file)
        adjusted_limit = math.ceil(limit/self.gen_batch_size)
        batch_amt = 0
        batches = self.data_generator_cnn(batch_size=self.gen_batch_size,filename=self.train_file)
        for batch in  tqdm(batches, total=adjusted_limit, desc="Creating Scalar"):
            if batch_amt > (limit-1):
                break
            else:
                #if end up needing matrix scaling, this line should be replaced with splitting matrix into n col name and value for 64 new columns per 8x8
                #matrixScaler.partial_fit(batch[2])
                scaler.partial_fit(batch[1])
                batch_amt += self.gen_batch_size
        self.init_scaler(scaler=scaler)
        #self.init_scaler(scaler=matrixScaler,scalarFile=self.matrixScalerFile)
        return scaler
    
    def data_generator_cnn(self, batch_size,filename):
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                X, Y, matrix_data = self.clean_data_cnn(chunk)
                Y = np.array(Y)
                #output = np.reshape(Y,(-1,1))
                output = Y
                yield (matrix_data, X, output )

    def scaled_data_generator_cnn(self, batch_size,filename):
        scaler = self.load_scaler()
        #matrixScaler = self.load_scaler(scalarFile=self.matrixScalerFile)
        while True:  # Loop indefinitely
            data = pd.read_csv(filename, chunksize=batch_size)
            for chunk in data:
                    X, Y, matrixData = self.clean_data_cnn(chunk)
                    X_scaled = scaler.transform(X)
                    matrixData_scaled = matrixData.applymap(reshape_to_matrix)
                    output_matrices = np.stack(matrixData_scaled.apply(lambda row: np.stack(row, axis=-1), axis=1).to_numpy())

                    #matrixData_scaled = matrixScaler.transform(matrixData)
                    Y = np.array(Y)
                    #output = np.reshape(Y,(-1,1))
                    output = Y
                    yield ((output_matrices,X_scaled), output)

    def get_cnn_shape(self):
        matrix_channels = 0
        metadata_columns = 0
        batch = next(self.data_generator_cnn(batch_size=self.gen_batch_size,filename=self.train_file))
        matrix_channels = len(batch[0].columns)
        metadata_columns = len(batch[1].columns)
        matrix_shape = (8,8,matrix_channels)
        metadata_shape = (metadata_columns,)
        self.cnn_shape = [matrix_shape,metadata_shape]
        return [matrix_shape,metadata_shape]
    
    def dataset_from_generator_cnn(self,filename,batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size
        shapes = self.cnn_shape
        matrix_shape = shapes[0]
        meta_shape = shapes[0]
        dataset = tf.data.Dataset.from_generator(
            lambda: self.scaled_data_generator_cnn(filename=filename, batch_size=batch_size),
            output_types=((tf.float32, tf.float32),tf.float32),  # Update these types based on your data
            output_shapes=(([None, 0, matrix_shape[1], matrix_shape[2]], [None, meta_shape[0]]), [None, 3])  # Update shapes based on your data
        )
        return dataset
                    
def string_to_array(s):
    # Safely evaluate the string as a Python literal (list of lists in this case)
    return np.array(ast.literal_eval(s))

def flat_string_to_array(s):
    cleaned_string = s.strip('[]\'').replace('\n', '')
    integer_array = np.array([np.float32(num) for num in cleaned_string.split()])
    return integer_array

def reshape_to_matrix(cell):
    return np.array(cell).reshape(8, 8)