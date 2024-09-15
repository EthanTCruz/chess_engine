import chess
import pandas as pd
import numpy as np
import chess
import torch
from chess_engine.src.model.config.config import Settings
from  chess_engine.src.model.classes.cnn_bb_scorer import boardCnnEval
from chess_engine.src.model.classes.torch_model import ModelOperator
from chess_engine.src.model.classes.mongo_functions import create_cnn_input

class board_analyzer():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        self.mdp = ModelOperator()
        self.mdp.load_model(model_path=s.torch_model_file)
        self.evaluator = boardCnnEval()

        self.means = np.load(s.np_means_file)
        self.stds = np.load(s.np_stds_file)
        # self.nnPredictionsCSV = s.nnPredictionsCSV


        # if "neuralNet" not in kwargs:
        #     raise Exception("No neural net supplied")
        # else:
        #     self.nn = kwargs["neuralNet"]
        


    def use_model(self,board:chess.Board):

        bitboards,metadata = self.evaluator.get_board_scores_applied(board=board)
        
        metadata = (metadata - self.means) / self.stds
        features = metadata.shape[1]
        metadata = metadata.reshape(1,1,features)
        n_bb = bitboards.shape[0]
        bitboards = bitboards.reshape(1,n_bb,8,8)
        metadata = torch.tensor(metadata, dtype=torch.float32)
        bitboards = torch.tensor(bitboards, dtype=torch.float32)
        bb = self.evaluator.get_board_scores(board=board)['positions_data']
        bb_o = create_cnn_input(bb)
        score = self.mdp.predict_single_example(bitboards=bitboards,metadata=metadata)

        # white = 0, black = 1, stalemate = 2
        return score

        
    

