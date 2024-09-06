import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from game_analyzer import game_analyzer
import numpy as np
from redis_populator import populator
import redis
import os
import chess


class principal_component_analysis():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,**kwargs):

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

        if "redis_client" not in kwargs:
            self.r = redis.Redis(host='localhost', port=6379)
        else:
            self.r = kwargs["redis_client"]
        
        if "ModelFilePath" or "ModelFilename" not in kwargs:
            ModelFilePath="./"
            ModelFilename="chess_model"
            self.ModelFile = f"{ModelFilePath}{ModelFilename}"
        else:
            self.ModelFile = f"{kwargs['ModelFilePath']}{kwargs['ModelFilename']}"
        
        if "filename" not in kwargs:
            self.filename = "data.csv"
        else:
            self.filename = kwargs["filename"]

        if "player" not in kwargs:
            self.game_analyzer_obj = game_analyzer(output_file=self.filename,player="NA")
        else:
            self.game_analyzer_obj = game_analyzer(output_file=self.filename,player=kwargs["player"])

        if "test_size" not in kwargs:
            self.test_size=.2
        else:
            self.test_size=kwargs["test_size"]
        
        if "random_state" not in kwargs:
            self.random_state=42
        else:
            self.random_state = kwargs["random_state"]


    def clean_data(self,data):
        # Remove the moves column, as it's not a useful feature for the neural network
        data = data.drop(columns=['moves(id)'])
        b_count = data[data['w/b'] == 'b'].shape[0]
        w_count = data[data['w/b'] == 'w'].shape[0]

        # calculate number of 'w' rows to keep (which is 3/4 of 'w' count or 'b' count whichever is lower)
        w_keep = min(int(w_count * 0.75), b_count)

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
        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=self.random_state)

        # Scale the feature data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, Y_train, Y_test

    
    def create_and_evaluate_model(self):

        data = pd.read_csv(self.filename)
        X,Y = self.clean_data(data=data)
        
        X_train, X_test, Y_train, Y_test = self.partition_and_scale(X=X,Y=Y)
        # Create a simple neural network model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, Y_train, batch_size=32, epochs=self.epochs, validation_split=0.2)
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, Y_test)

        tf.keras.models.save_model(model=model,filepath=self.ModelFile)

        print("Test loss:", loss)
        print("Test accuracy:", accuracy)
        return model
    
    def process_redis_boards(self):        
        self.game_analyzer_obj.process_redis_boards()
        data = pd.read_csv(self.filename)

        X,moves = self.clean_prediction_data(data=data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = tf.keras.models.load_model(filepath=self.ModelFile)

        predictions = model.predict(X)
        moves['predictions'] = predictions
        return(moves)

    def score_board(self,board_key):
        self.game_analyzer_obj.process_single_board(board_key=board_key)
        
        data = pd.read_csv(self.filename)

        X,Y = self.clean_data(data=data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = tf.keras.models.load_model(filepath=self.ModelFile)

        prediction = float(model.predict(X))

        return(prediction)

    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        
    def send_move_scores_to_redis(self,board: chess.Board):
        self.game_analyzer_obj.process_redis_boards()
        #Line above is needs optomizing
        data = pd.read_csv(self.filename)
        checkmates = data[data["checkmate"].isin([1,-1])]
        data = data[~data['checkmate'].isin([1,-1])]
        X,moves = self.clean_prediction_data(data=data)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        model = tf.keras.models.load_model(filepath=self.ModelFile)

        data['predictions'] = model.predict(X)
        checkmates['predictions'] = checkmates.loc[:,'checkmate']
        data = pd.concat([data,checkmates])
        self.r.flushdb()
        for index, row in data.iterrows():
            move = row['moves(id)']
            score = float(row['predictions'])
            self.r.set(move,score)
        # for i in range(0,len(moves)-1):
        #         move = moves[i]
        #         score = float(data['predictions'][i])
        #         self.r.set(move,score)






