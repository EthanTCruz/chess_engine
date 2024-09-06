import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from chess_engine.src.model.classes.game_analyzer import game_analyzer
import numpy as np
import chess
from chess_engine.src.model.config.config import Settings
import random
from joblib import dump, load
from chess_engine.src.model.classes.gcp_operations import upload_blob, download_blob
import math
from chess_engine.src.model.classes.dataGenerator import data_generator,flat_string_to_array,reshape_to_matrix
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Concatenate
from tensorflow.keras.models import Model


class neural_net():

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



            

        self.dataGenerator = data_generator(**kwargs)
        self.gcp_creds = s.GOOGLE_APPLICATION_CREDENTIALS
        self.bucket_name = s.BUCKET_NAME
        self.saveToBucket = s.saveToBucket



    def clean_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        data = data.drop(columns=['moves(id)'])
        b_count = data[data['w/b'] == 'b'].shape[0]
        w_count = data[data['w/b'] == 'w'].shape[0]

        # calculate number of 'w' rows to keep (which is 3/4 of 'w' count or 'b' count whichever is lower)
        w_keep = min(w_count, b_count)

        # get indices of 'w' rows
        w_indices = data[data['w/b'] == 'w'].index

        # choose random subset of 'w' indices to keep
        w_indices_keep = np.random.choice(w_indices, w_keep, replace=False)

        # get all 'b' indices
        b_indices = data[data['w/b'] == 'b'].index

        # combine 'w' indices to keep and all 'b' indices
        new_indices = np.concatenate([w_indices_keep, b_indices])

        # filter dataframe to these indices
        data = data.loc[new_indices]
                
        # Encode the target variable (w/b) as 0 or 1
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)
        # One-hot encode the 'game time' feature

        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])
        Y = data[self.target_feature]

        return X,Y


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
    

    
    def clean_prediction_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        Y = data['moves(id)']
        data = data.drop(columns=['moves(id)'])

        # Encode the target variable (w/b) as 0 or 1
        data[self.target_feature] = data[self.target_feature].apply(lambda x: 1 if x == 'w' else 0)
        # One-hot encode the 'game time' feature

        # Split the data into features and target
        X = data.drop(columns=[self.target_feature])


        return X,Y



    def partition_and_scale(self,X,Y):
        random.seed(3141)
        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

        # Scale the feature data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        self.init_scaler(scaler=scaler)
        return X_train, X_test, Y_train, Y_test
    
        
    def get_row_count(self,filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)


        
    def create_and_evaluate_model_batch(self):
        self.dataGenerator.initialize_datasets()
        batch_size = self.batch_size  # Batch size

        validation_batch = self.batch_size
        validation_samples = self.get_row_count(filename=self.validation_file)
        validation_steps = math.ceil(validation_samples/validation_batch)
        shape = self.dataGenerator.get_shape()


        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Dense(128, activation='relu', input_shape=(shape,)),
            tf.keras.layers.BatchNormalization(),

            # Hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Dropout for regularization
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),

            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


        train_dataset = self.dataGenerator.dataset_from_generator(
                filename=self.train_file,
                batch_size=batch_size)
        validation_dataset = self.dataGenerator.dataset_from_generator(
                filename=self.validation_file,
                batch_size=batch_size)
        # Train the model
        model.fit(train_dataset,
                  steps_per_epoch=batch_size,
                  epochs=self.epochs, 
                  validation_data=validation_dataset,
                  validation_steps=validation_steps)
        # Evaluate the model on the test set

        test_dataset = self.dataGenerator.dataset_from_generator(
                filename=self.test_file,
                batch_size=batch_size,)
        
        test_size = self.dataGenerator.test_dataset_size(filename=self.test_file)
        steps = math.ceil((test_size - 1) / batch_size)
        loss, accuracy = model.evaluate(test_dataset,steps=steps)

        tf.keras.models.save_model(model=model,filepath=self.ModelFile)

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        destination_blob_name = f"models/{timestamp}-{self.ModelFilename}"
        upload_blob(bucket_name=self.bucket_name,
                    source_file_name=self.ModelFile,
                    destination_blob_name=destination_blob_name)
        
        self.reload_model()
        return loss,accuracy
        #return model
    
    def create_and_evaluate_cnn_model_batch(self):
        #self.dataGenerator.initialize_cnn_datasets()
        batch_size = self.batch_size  # Batch size
        training_size = self.get_row_count(filename=self.train_file)
        steps_per_epoch = math.ceil(training_size/batch_size)
        validation_batch = self.batch_size
        validation_samples = self.get_row_count(filename=self.validation_file)
        validation_steps = math.ceil(validation_samples/validation_batch)
        shape = self.dataGenerator.get_shape_cnn()
        #test = next(self.dataGenerator.scaled_data_generator_cnn(batch_size=self.gen_batch_size,filename=self.train_file))
        
        matrix_shape =  Input(shape=shape[0])
        metadata_shape = Input(shape=shape[1])
        # Convolutional layers for chessboard
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(matrix_shape)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
        flattened = Flatten()(conv2)

        # Dense layers for metadata
        meta_dense = Dense(64, activation='relu')(metadata_shape)

        # Combine outputs
        combined = Concatenate()([flattened, meta_dense])

        # Additional dense layers
        dense1 = Dense(128, activation='relu')(combined)
        dense2 = Dense(64, activation='relu')(dense1)

        # Output layer
        output = Dense(3, activation='softmax')(dense2)

        model = Model(inputs=[matrix_shape, metadata_shape], outputs=output)

        train_dataset = self.dataGenerator.dataset_from_generator_cnn(
                filename=self.train_file,
                batch_size=batch_size)
        validation_dataset = self.dataGenerator.dataset_from_generator_cnn(
                filename=self.validation_file,
                batch_size=batch_size)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        
        # Train the model
        history = model.fit(train_dataset,
                  steps_per_epoch=steps_per_epoch,
                  epochs=self.epochs, 
                  validation_data=validation_dataset,
                  validation_steps=validation_steps)
        # Evaluate the model on the test set

        test_dataset = self.dataGenerator.dataset_from_generator_cnn(
                filename=self.test_file,
                batch_size=batch_size,)
        
        test_size = self.dataGenerator.test_dataset_size(filename=self.test_file)
        steps = math.ceil((test_size - 1) / batch_size)
        loss, accuracy = model.evaluate(test_dataset,steps=steps)

        model.save(filepath=self.ModelFile)
        
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)

        if self.saveToBucket:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            destination_blob_name = f"models/{timestamp}-{self.ModelFilename}"
            upload_blob(bucket_name=self.bucket_name,
                        source_file_name=self.ModelFile,
                        destination_blob_name=destination_blob_name)
            
        self.reload_model()
        return loss,accuracy
    
        #return model

    def create_and_evaluate_model(self):

        data = pd.read_csv(self.filename)
        X,Y = self.clean_data(data=data)

        X_train, X_test, Y_train, Y_test = self.partition_and_scale(X=X,Y=Y)


        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.BatchNormalization(),

            # Hidden layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),  # Dropout for regularization
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),

            # Output layer
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, Y_test)



        tf.keras.models.save_model(model=model,filepath=self.ModelFile)

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        self.reload_model()


        return model
    



    def score_board(self,board: chess.Board):
        fen = board.fen()
        #self.game_analyzer_obj.process_single_fen(fen=fen)
        
        #data = pd.read_csv(self.filename)
        data = self.game_analyzer_obj.process_single_fen_non_csv(fen=fen)
        X,Y = self.clean_prediction_data(data=data)
        scaler =  self.load_scaler()
        X = scaler.fit_transform(X)



        prediction = float(self.model.predict(X))

        return(prediction)

    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        


    def send_move_scores_to_csv(self,total_moves: dict = {}):

        
        self.game_analyzer_obj.process_boards(total_moves=total_moves)


        #Line above is needs optomizing
        data = pd.read_csv(self.filename)

        checkmates = data[data["checkmate"] == 1]
        data = data[data['checkmate'] != 1]
        if len(data) > len(data[data["w/b"] == 's']):
            data = data[~(data["w/b"] == 's')]

        if len(data) > len(data[data["can be drawn"] == 1]):
            data = data[~(data["can be drawn"] == 1)]
        if len(data) > 0:
            data.reset_index(drop=True, inplace=True)
            X,moves = self.clean_prediction_data(data=data)
            scaler =  self.load_scaler()
            X = scaler.fit_transform(X)
            data['prediction'] = self.model.predict(X)

        checkmates['prediction'] = np.where(checkmates['w/b'] == 'b', 
                                     checkmates['checkmate'] * 0, 
                                     checkmates['checkmate'])
        data = pd.concat([data,checkmates])
        data.reset_index(drop=True, inplace=True)
        #error: should fix to be specific db

        output_file = self.nnPredictionsCSV
        data.to_csv(output_file, index=False)

    def score_moves(self,total_moves: dict = {}):
     
        data = self.game_analyzer_obj.process_boards_non_csv(total_moves=total_moves)

        checkmates = data[data["checkmate"] == 1]
        data = data[data['checkmate'] != 1]
        if len(data) > len(data[data["w/b"] == 's']):
            data = data[~(data["w/b"] == 's')]

        if len(data) > len(data[data["can be drawn"] == 1]):
            data = data[~(data["can be drawn"] == 1)]
        if len(data) > 0:
            data.reset_index(drop=True, inplace=True)
            X,moves = self.clean_prediction_data(data=data)
            scaler =  self.load_scaler()
            X = scaler.fit_transform(X)
            data['prediction'] = self.model.predict(X)

        checkmates['prediction'] = np.where(checkmates['w/b'] == 'b', 
                                     checkmates['checkmate'] * 0, 
                                     checkmates['checkmate'])
        data = pd.concat([data,checkmates])
        data.reset_index(drop=True, inplace=True)
        #error: should fix to be specific db

        return data



    def clean_prediction_data_cnn(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        Y = data['moves(id)']
        data = data.drop(columns=['moves(id)'])



        # Split the data into features and target
        X = data.drop(columns=self.target_feature)
        
        matrix_data = pd.DataFrame()
        for col in data.columns:
            if col.endswith('positions'):
                matrix_data[col] = data[col]

        # DataFrame without 'positions' columns
        non_positions_data = data.drop(columns=matrix_data.columns)

        X = non_positions_data.drop(columns=self.target_feature)

        return X, matrix_data, Y 



 
    def score_moves_cnn(self,total_moves: dict = {}):
     
        data = self.game_analyzer_obj.process_boards_non_csv_cnn(total_moves=total_moves)

        checkmates = data[(data["white mean"] == 1) | (data["black mean"] == 1)]

        data = data[~data['moves(id)'].isin(checkmates['moves(id)'])]
        if len(data) > len(data[data["stalemate mean"] == 1]):
            data = data[~(data["stalemate mean"] == 1)]

        if len(data) > 0:
            data.reset_index(drop=True, inplace=True)
            X,matrixData,moves = self.clean_prediction_data_cnn(data=data)
            #matrixData_scaled = matrixData.applymap(reshape_to_matrix)
            reshaped_arrays = []

            for col in matrixData.columns:
                # Reshape each column into 8x8 and convert to float32 tensor
                reshaped_col = np.stack(matrixData[col].apply(lambda x: np.array(x).reshape(8, 8)), axis=0)
                reshaped_col_tensor = tf.convert_to_tensor(reshaped_col, dtype=tf.float32)
                reshaped_arrays.append(reshaped_col_tensor)

            # Stack along the last axis to create a [None, 8, 8, 14] tensor
            final_tensor = tf.stack(reshaped_arrays, axis=-1)
            #tensor_data = tf.convert_to_tensor(matrixData_scaled, dtype=tf.float32)

            # Reshape the tensor to the desired shape
            # The -1 in reshape allows the batch size to be dynamically computed
            #reshaped_data = tf.reshape(tensor_data, [-1, 8, 8, 14])
            scaler =  self.load_scaler()
            X = scaler.fit_transform(X)
            predictions = self.model.predict((final_tensor,X))
            if len(data) == len(predictions):
                data['prediction'] = [pred for pred in predictions]
            else:
                raise ValueError("Number of predictions does not match number of rows in the DataFrame.")


        checkmates['prediction'] = checkmates.apply(lambda row: [row['white mean'], row['black mean'], row['stalemate mean']], axis=1)

        data = pd.concat([data,checkmates])
        data = data[['moves(id)', 'prediction']]
        data.reset_index(drop=True, inplace=True)
        #error: should fix to be specific db
        data['stalemate'] = data['prediction'].apply(lambda x: x[2])
        data['black'] = data['prediction'].apply(lambda x: x[1])
        data['white'] = data['prediction'].apply(lambda x: x[0])
        data.drop(columns=['prediction'])

        return data

    def score_board_cnn(self,board: chess.Board):

        #self.game_analyzer_obj.process_single_fen(fen=fen)
        
        #data = pd.read_csv(self.filename)
        data = self.game_analyzer_obj.process_single_fen_non_csv_cnn(board=board)


        X,matrixData,moves = self.clean_prediction_data_cnn(data=data)
        #matrixData_scaled = matrixData.applymap(reshape_to_matrix)
        reshaped_arrays = []

        for col in matrixData.columns:
            # Reshape each column into 8x8 and convert to float32 tensor
            reshaped_col = np.stack(matrixData[col].apply(lambda x: np.array(x).reshape(8, 8)), axis=0)
            reshaped_col_tensor = tf.convert_to_tensor(reshaped_col, dtype=tf.float32)
            reshaped_arrays.append(reshaped_col_tensor)

        final_tensor = tf.stack(reshaped_arrays, axis=-1)

        X = self.scaler.fit_transform(X)
        predictions = self.model.predict((final_tensor,X))
        if len(data) == len(predictions):
            data['prediction'] = [pred for pred in predictions]
        else:
            raise ValueError("Number of predictions does not match number of rows in the DataFrame.")



        data = data[['moves(id)', 'prediction']]
        data.reset_index(drop=True, inplace=True)
        #error: should fix to be specific db
        data['stalemate'] = data['prediction'].apply(lambda x: x[2])
        data['black'] = data['prediction'].apply(lambda x: x[1])
        data['white'] = data['prediction'].apply(lambda x: x[0])
        data.drop(columns=['prediction'])



        return(data)