import pandas as pd

import chess
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.MCTS import mcts
from chess_engine.src.model.classes.pretorch_files.potential_board_populator import populator



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
        baseBoard = chess.Board()
        baseDepth = 1
        self.populator = populator(board=baseBoard,score_depth=baseDepth)



    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)

    
    def average_and_stdv_scores(self,board: chess.Board):
        legal_moves = self.get_legal_moves(board=board)
        move_stats = {}
        for move in legal_moves:
            move_scores = []
            data = pd.read_csv(self.nnPredictionsCSV)



            if len(move_scores) > 0:  # or len(move_scores) > 0 if it's a list
                mean = data["prediction"].mean()
                stdv = data["prediction"].std()
            else:
                mean = 0.5
                stdv = 0
            move_stats[move] = [mean, stdv]

        # Convert move_stats to a pandas DataFrame
        df = pd.DataFrame.from_dict(move_stats, orient='index', columns=['Mean', 'Standard Deviation'])

        # Set the move as the index
        df.index.name = 'Move'
        
        return df

    def highest_average_move(self,board: chess.Board,player: str = ""):
        if player == "":
            if board.turn:
                player = 'w'
            else:
                player = 'b'
        move_stats = self.average_and_stdv_scores(board)

        if player == 'w':
            move = move_stats["Mean"].idxmax()
        else:
            move = move_stats["Mean"].idxmin()
        return move

        



    def use_model(self,board: chess.Board = chess.Board(),score_depth: int = 1,percentile: float = 0.75):
        
        self.populator.set_depth_and_board(depth=score_depth,board=board)
        self.populator.get_all_moves()
        moves = self.populator.return_total_moves()

        scores = self.nn.score_moves(total_moves=moves)

        move = self.mcts_best(board=board,iterations=100,max_depth=500,percentile=percentile,move_scores=scores)

        return move
    
            
    def get_best_moves(self, board: chess.Board,move_scores: pd.DataFrame, percentile: float = 0.5):
        mean = move_scores["prediction"].mean()
        if board.turn:
            percentile = move_scores["prediction"].quantile(percentile)
            # filtered_moves = move_scores[move_scores['prediction'] > percentile]
            # filtered_moves = filtered_moves[filtered_moves['prediction'] >= mean]
            filtered_moves = move_scores[move_scores['prediction'] >= mean]
            sorted_series = filtered_moves.sort_values(by='prediction',ascending=False)
            mate_moves = filtered_moves[filtered_moves['prediction'] == 1]
        else:
            quantile = 1 - percentile
            percentile = move_scores["prediction"].quantile(quantile)
            # filtered_moves = move_scores[move_scores['prediction'] < percentile]
            # filtered_moves = filtered_moves[filtered_moves['prediction'] <= mean]
            filtered_moves = move_scores[move_scores['prediction'] <= mean]
            sorted_series = filtered_moves.sort_values(by='prediction',ascending=True)
            mate_moves = filtered_moves[filtered_moves['prediction'] == 0]

        if len(mate_moves) > 0:
            sorted_series = mate_moves

        index_list = list(sorted_series["moves(id)"].values)
        return index_list
    
    def get_all_moves(self, board: chess.Board,move_scores: pd.DataFrame, percentile: float = 0.5):


        sorted_series = move_scores.sort_values(by='prediction',ascending=True)


        index_list = list(sorted_series["moves(id)"].values)
        return index_list
        
    def mcts_best(self,board: chess.Board,move_scores: pd.DataFrame,iterations: int = 10,max_depth: int = 10,percentile: float = 0.5):
        preferred_moves = self.get_all_moves(board=board,percentile=percentile,move_scores=move_scores)

        #preferred_moves = self.get_best_moves(board=board,percentile=percentile,move_scores=move_scores)
        
        move = self.ms.mcts_best_move(board=board,preferred_moves=preferred_moves,iterations=iterations, max_depth=max_depth)
        return move
    
    def use_model_cnn(self,board: chess.Board = chess.Board(),score_depth: int = 1,percentile: float = 0.75):
        
        self.populator.set_depth_and_board(depth=score_depth,board=board)
        self.populator.get_all_moves()
        moves = self.populator.return_total_moves()

        scores = self.nn.score_moves_cnn(total_moves=moves)

        move = self.mcts_best_cnn(board=board,iterations=100,max_depth=500,percentile=percentile,move_scores=scores)

        return move
    
    def get_all_moves_cnn(self, white: bool,move_scores: pd.DataFrame, percentile: float = 0.5):

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
    
    def mcts_best_cnn(self,board: chess.Board,move_scores: pd.DataFrame,iterations: int = 10,max_depth: int = 10,percentile: float = 0.5):
        preferred_moves = self.get_all_moves_cnn(white=board.turn,percentile=percentile,move_scores=move_scores)


        move = self.ms.mcts_best_move(board=board,preferred_moves=preferred_moves,iterations=iterations, max_depth=max_depth,cnn=True)
        return move