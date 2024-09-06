import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def configure_gpu_memory_growth():
    gpus= tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

configure_gpu_memory_growth()

from chess_engine.src.model.classes.pretorch_files.cnn_game_analyzer import game_analyzer
import numpy as np
import chess
from chess_engine.src.model.config.config import Settings
import random
from joblib import dump, load
from chess_engine.src.model.classes.gcp_operations import upload_blob, download_blob
import math
from chess_engine.src.model.classes.cnn_dataGenerator import data_generator
from chess_engine.src.model.classes.cnn_recordGenerator import record_generator
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model

class convolutional_neural_net():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        
        self.nnPredictionsCSV = s.nnPredictionsCSV

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


        
        if "ModelFilePath" or "ModelFilename" not in kwargs:
            self.ModelFilePath= s.ModelFilePath
            self.ModelFilename = s.ModelFilename
            self.ModelFile = f"{self.ModelFilePath}{self.ModelFilename}"
        else:
            self.ModelFile = f"{kwargs['ModelFilePath']}{kwargs['ModelFilename']}"
        
                
        if "scalerFile" not in kwargs:
            self.scalerFile = s.scaler_weights
        else:
            self.scalerFile = kwargs["scalerFile"]

        #fix later for now just load
        if "trainModel" in kwargs:
            if kwargs["trainModel"]:
                pass
            else:
                self.model = tf.keras.models.load_model(filepath=self.ModelFile)

        else:
            self.model = tf.keras.models.load_model(filepath=self.ModelFile)

        if "filename" not in kwargs:
            self.filename = "data.csv"
        else:
            self.filename = kwargs["filename"]


        self.game_analyzer_obj = game_analyzer(output_file=self.filename)



        if "test_size" not in kwargs:
            self.test_size=.2
        else:
            self.test_size=kwargs["test_size"]
        
        if "random_state" not in kwargs:
            self.random_state=42
        else:
            self.random_state = kwargs["random_state"]


        if "trainingFile" not in kwargs:
            self.train_file = s.trainingFile
        else:
            self.train_file  = kwargs["trainingFile"]

        if "testingFile" not in kwargs:
            self.test_file = s.testingFile
        else:
            self.test_file = kwargs["testingFile"]


        if "nnGenBatchSize" not in kwargs:
            self.gen_batch_size = s.nnGenBatchSize
        else:
            self.gen_batch_size = kwargs["nnGenBatchSize"]


        if "copy_file" not in kwargs:
            self.copy_data = s.copy_file
        else:
            self.copy_data  = kwargs["copy_file"]


        if "validationFile" not in kwargs:
            self.validation_file = s.validationFile
        else:
            self.validation_file = kwargs["validationFile"]            

        if "nnValidationSize" not in kwargs:
            self.validation_size = s.nnValidationSize
        else:
            self.validation_size =  kwargs["nnValidationSize"]
 

        self.dataGenerator = record_generator(**kwargs)
        # self.dataGenerator = data_generator(**kwargs)
        self.gcp_creds = s.GOOGLE_APPLICATION_CREDENTIALS
        self.bucket_name = s.BUCKET_NAME
        self.saveToBucket = s.saveToBucket
        self.log_dir = s.nnLogDir
        self.checkpoints = s.nnModelCheckpoint
        self.recordsDataFileTrain = s.recordsDataTrain
        self.recordsDataFileTest = s.recordsDataTest
        self.recordsDataFileValidation = s.recordsDataValidation


    def load_scaler(self):
        scaler = load(self.scalerFile)
        self.scaler = scaler
        return scaler
    
    def init_scaler(self,scaler:StandardScaler):
        dump(scaler, self.scalerFile)
        return 0
    
    #for to reload model on self play data, work on later
    def reload_model(self,modelFile: str = ""):
        if modelFile == "":
            modelFile = self.ModelFile
        self.model = tf.keras.models.load_model(filepath=modelFile)
        return 0
    


        
    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)

    def create_model(self,shapes_tuple):
        # Input layers
        # bitboard_shape = Input(shape=shapes_tuple[0][0])
        # metadata_shape = Input(shape=shapes_tuple[0][1])
        bitboard_shape = Input(shape=shapes_tuple[0])
        metadata_shape = Input(shape=shapes_tuple[1])
        # Process bitboards
        bitboard_input = Flatten()(bitboard_shape)
        x = Dense(512, activation='relu')(bitboard_input)

        # Process metadata
        y = Dense(64, activation='relu')(metadata_shape)

        # Combine bitboard and metadata features
        combined = Concatenate()([x, y])

        # Hidden layers
        z = Dense(256, activation='relu')(combined)
        z = Dense(128, activation='relu')(z)

        # Output layer
        output = Dense(3, activation='softmax')(z)

        model = Model(inputs=[bitboard_shape, metadata_shape], outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def calc_step_sizes(self):
        batch_size = self.batch_size  # Batch size
        
        # training_size = self.get_row_count(filename=self.train_file)
        training_size = count_records(self.recordsDataFileTrain)
        steps_per_epoch = math.ceil(training_size/batch_size)
        validation_batch = self.batch_size
        # validation_samples = self.get_row_count(filename=self.validation_file)
        validation_samples = count_records(self.recordsDataFileValidation)
        validation_steps = math.ceil(validation_samples/validation_batch)
        return steps_per_epoch, validation_steps,batch_size



    def create_and_evaluate_model(self):
        tensorboard_callback =  tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, 
                                                               histogram_freq=1,
                                                                write_graph=True,
                                                                write_images=True)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoints,
                                                 save_weights_only=True,
                                                 verbose=1)

        #tensorboard --logdir='/home/user/chess_engine/logs/train'
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) > 0:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        #self.dataGenerator.initialize_cnn_datasets()
        steps_per_epoch,validation_steps,batch_size = self.calc_step_sizes()
        shape = self.dataGenerator.get_shape()

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            model = self.create_model(shapes_tuple=shape)




            
            # Train the model
            history = model.fit(self.dataGenerator.train_data,
                    steps_per_epoch=steps_per_epoch,
                    epochs=self.epochs, 
                    validation_data=self.dataGenerator.validation_data,
                    validation_steps=validation_steps,
                    callbacks=[tensorboard_callback,cp_callback])
            # Evaluate the model on the test set

            
            test_size = count_records(self.recordsDataFileTest)
            steps = math.ceil((test_size - 1) / batch_size)
            loss, accuracy = model.evaluate(self.dataGenerator.test_data,steps=steps)

            model.save(filepath=self.ModelFile)
            
            print("Test loss:", loss)
            print("Test accuracy:", accuracy)

            if self.saveToBucket:
                self.save_model_to_bucket()
                
            self.reload_model()
            return loss,accuracy
    
        #return model


    def save_model_to_bucket(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination_blob_name = f"models/{timestamp}-{self.ModelFilename}"
        upload_blob(bucket_name=self.bucket_name,
                    source_file_name=self.ModelFile,
                    destination_blob_name=destination_blob_name)



    def clean_prediction_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        Y = data['moves(id)']
        data = data.drop(columns=['moves(id)'])
        X = data.drop(columns=self.target_feature)

        matrix_data = self.separate_positions_matrices(data=data)

        non_positions_data = data.drop(columns=matrix_data.columns)
        X = non_positions_data.drop(columns=self.target_feature)

        return X, matrix_data, Y 

    def separate_positions_matrices(self,data: pd.DataFrame):
        matrix_data = pd.DataFrame()
        for col in data.columns:
            if col.endswith('positions'):
                matrix_data[col] = data[col]
        return matrix_data

 
    def score_moves(self,total_moves: dict = {}):
     
        data = self.game_analyzer_obj.process_boards(total_moves=total_moves)

        checkmates = data[(data["white mean"] == 1) | (data["black mean"] == 1)]

        data = data[~data['moves(id)'].isin(checkmates['moves(id)'])]
        if len(data) > len(data[data["stalemate mean"] == 1]):
            data = data[~(data["stalemate mean"] == 1)]

        if len(data) > 0:
            data.reset_index(drop=True, inplace=True)
            X,matrixData,moves = self.clean_prediction_data(data=data)

            final_tensor = self.df_matrices_to_tensor(matrixData=matrixData)

            scaler =  self.load_scaler()
            X = scaler.fit_transform(X)
            predictions = self.model.predict((final_tensor,X))
            if len(data) == len(predictions):
                data['prediction'] = [pred for pred in predictions]
            else:
                raise ValueError("Number of predictions does not match number of rows in the DataFrame.")

        checkmates['prediction'] = checkmates.apply(lambda row: [row['white mean'], row['black mean'], row['stalemate mean']], axis=1)
        data = pd.concat([data,checkmates])
        data = self.df_results_cleanse(data=data)

        return data

    def df_matrices_to_tensor(self,matrixData:pd.DataFrame):
        reshaped_arrays = []

        for col in matrixData.columns:
            # Reshape each column into 8x8 and convert to float32 tensor
            reshaped_col = np.stack(matrixData[col].apply(lambda x: np.array(x).reshape(8, 8)), axis=0)
            reshaped_col_tensor = tf.convert_to_tensor(reshaped_col, dtype=tf.float32)
            reshaped_arrays.append(reshaped_col_tensor)

        final_tensor = tf.stack(reshaped_arrays, axis=-1)

        return final_tensor

    def score_board(self,board: chess.Board):
        data = self.game_analyzer_obj.process_single_board(board=board)
        X,matrixData,moves = self.clean_prediction_data(data=data)

        final_tensor = self.df_matrices_to_tensor(matrixData=matrixData)
        X = self.scaler.fit_transform(X)
        predictions = self.model.predict((final_tensor,X))

        if len(data) == len(predictions):
            data['prediction'] = [pred for pred in predictions]
        else:
            raise ValueError("Number of predictions does not match number of rows in the DataFrame.")

        data = self.df_results_cleanse(data=data)



        return(data)
    
    def df_results_cleanse(self,data:pd.DataFrame):
        data = data.copy()
        data = data[['moves(id)', 'prediction']]
        data.reset_index(drop=True, inplace=True)
        #error: should fix to be specific db
        data['stalemate'] = data['prediction'].apply(lambda x: x[2])
        data['black'] = data['prediction'].apply(lambda x: x[1])
        data['white'] = data['prediction'].apply(lambda x: x[0])
        data.drop(columns=['prediction'])
        return data
    
def count_records(tfrecords_filename):
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecords_filename):
        count += 1
    return count