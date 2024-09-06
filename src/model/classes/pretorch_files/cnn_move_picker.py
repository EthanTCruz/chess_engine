import pandas as pd

import chess
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.MCTS import mcts
from chess_engine.src.model.classes.pretorch_files.potential_board_populator import populator
from chess_engine.src.model.classes.cnn_MCTS import mcts as mcts2


class move_picker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()

        self.nnPredictionsCSV = s.nnPredictionsCSV


        if "neuralNet" not in kwargs:
            raise Exception("No neural net supplied")
        else:
            self.nn = kwargs["neuralNet"]
        self.ms = mcts(neuralNet=self.nn)
        self.ms2 = mcts2(neuralNet=self.nn)
        baseBoard = chess.Board()
        baseDepth = 1
        self.populator = populator(board=baseBoard,score_depth=baseDepth)




    def use_model_timed(self,board: chess.Board = chess.Board(),time_limit:float = 10):


        move = self.ms2.mcst_timed(board=board,time_limit=time_limit)

        return move
        
    
    def use_model(self,board: chess.Board = chess.Board(),score_depth: int = 1,percentile: float = 0.75):
        
        self.populator.set_depth_and_board(depth=score_depth,board=board)
        self.populator.get_all_moves()
        moves = self.populator.return_total_moves()

        scores = self.nn.score_moves(total_moves=moves)

        move = self.mcts_best(board=board,iterations=1000,max_depth=100,percentile=percentile,move_scores=scores)

        return move
    
    def get_all_moves(self, white: bool,move_scores: pd.DataFrame, percentile: float = 0.5):

        stalemate_cutoff = move_scores['stalemate'].quantile((1-percentile))
        move_scores_filtered = move_scores[move_scores['stalemate'] <= stalemate_cutoff]
        if white:
            wins = move_scores_filtered[move_scores_filtered['white']==1].head(1)['moves(id)']
            if len(wins) > 0:
                return wins

            sorted_series = move_scores_filtered.sort_values(by='white',ascending=False)
        else:
            wins = move_scores_filtered[move_scores_filtered['black']==1].head(1)['moves(id)']
            if len(wins) > 0:
                return wins
            sorted_series = move_scores_filtered.sort_values(by='black',ascending=False)



        index_list = list(sorted_series["moves(id)"].values)
        return index_list
    
    def mcts_best(self,board: chess.Board,move_scores: pd.DataFrame,iterations: int = 10,max_depth: int = 10,percentile: float = 0.5):
        preferred_moves = self.get_all_moves(white=board.turn,percentile=percentile,move_scores=move_scores)


        move = self.ms.mcts_best_move(board=board,preferred_moves=preferred_moves,iterations=iterations, max_depth=max_depth,cnn=True)
        return move