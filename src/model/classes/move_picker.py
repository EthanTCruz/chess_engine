from chess_engine.src.model.classes.board_analyzer import board_analyzer
from chess_engine.src.model.classes.sqlite.dependencies import find_rollup_move

from chess_engine.src.model.classes.MCTS import mcts
import chess
import random
from math import sqrt

class move_picker():
    def __init__(self,ucb_constant:float = None,scores: list = [1.5,-1.5,0]) -> None:
        ba = board_analyzer()
        self.mcts = mcts(board_analyzer=ba,
                         ucb_constant=ucb_constant,
                         scores=scores)
        pass
        
    def get_best_move(self,board:chess.Board,
                      ucb_constant: float = sqrt(2),
                      scores: list = [1, -1, 0.5],
                      iterations:int = 100):
        random.seed(3141)
        self.mcts.set_ucb(ucb = ucb_constant)
        self.mcts.set_scores(scores=scores)
        return self.mcts.mcts_best_move(board=board,iterations=iterations)
    
    def get_rollup_move(self,board:chess.Board):
        return find_rollup_move(board=board)

    def full_engine_get_move(self,board:chess.Board,iterations: int = 128):
        move = self.get_rollup_move(board=board)
        if move != 0:
            print("rollup move")
            return move
        print("mcts move")
        move = self.get_best_move(board=board,iterations=iterations)
        print("move")
        return move

