import chess
import csv
import redis
import hiredis
import ezboard
import sys

DEPTH = 2



class populator():

    def __init__(self,depth,board: chess.Board,pgn = None,victor = None):
        self.DEPTH = depth
        self.r = redis.Redis(host='localhost', port=6379)
        self.board = board
        if pgn is not None:
            self.pgn = pgn
            if victor is not None:
                raise Exception("Need victor color")
            else:
                self.victor = victor

    def reset_and_fill_redis(self):
        self.r.flushdb()
        self.populate_redis(board=self.board,moves=self.get_legal_moves(board=self.board))



    def get_legal_moves(self,board: chess.Board):
        legal_moves=  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        

    def populate_redis(self,board: chess.Board, moves: list[str] = []):
        move_dict = {}
        for move in moves:
            try:
                board.push_uci(move)
                if len(board.move_stack) > self.DEPTH:
                    #implement nn here
                    value = "undefined"

                    if board.is_checkmate():
                        value = "checkmate"
                    if board.is_stalemate():
                        value = "stalemate"
                    self.r.set(f'{str([move.uci() for move in board.move_stack])}:{ board.fen()}',"undefined")
                else:
                    legal_moves = self.get_legal_moves(board)
                    if legal_moves:
                        sub_dict = self.populate_redis(board,moves=legal_moves)
                        move_dict.update(sub_dict)
            except ValueError:
                pass
            board.pop()
        return move_dict
    
        

