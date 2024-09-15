import sys
import os
import csv
import cowsay
import chess
import time
from math import log,sqrt,e,inf
import random
import numpy as np
import torch
sys.path.append('../')
from sqlalchemy.orm import  Session
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.sqlite.dependencies import (
    delete_all_game_positions,
    delete_all_rollup_game_positions,
    create_rollup_table,
    find_rollup_move,
    find_board_rollup)
from chess_engine.src.model.classes.pgn_processor import pgn_processor

from chess_engine.src.model.config.config import Settings

from chess_engine.src.model.classes.endgame import endgamePicker
from chess_engine.src.model.classes.mongo_functions import mongo_data_pipe
from chess_engine.src.model.classes.torch_model import ModelOperator
from chess_engine.src.model.classes.board_analyzer import board_analyzer
from chess_engine.src.model.classes.move_picker import move_picker


s = Settings()
ModelFilePath=s.ModelFilePath
ModelFilename=s.ModelFilename

pgn_file = s.pgn_file

epochs = s.nnEpochs
batch_size = s.nnBatchSize
test_size = s.nnTestSize

if s.useSamplePgn:
    pgn_file=s.samplePgn
# pgn_file=s.samplePgn



# mp = cnn_move_picker(neuralNet=nn)


mdp = mongo_data_pipe()

# ba = board_analyzer()

# mp = move_picker()



def main():
    # test_speeds()
    # pgn_to_db()
    full_data_to_ml()
    # initialize_collections()
    return 0







def get_data(pgn_file = pgn_file,db: Session = SessionLocal()):
    delete_all_game_positions(db = db)
    pgn_obj = pgn_processor(pgn_file=pgn_file)
    pgn_obj.pgn_fen_to_sqlite()

def preprocess_data():
    delete_all_rollup_game_positions()
    create_rollup_table(yield_size=500,db=SessionLocal())

def process_data():
    mdp.open_connections()
    
    mdp.initialize_data(batch_size=1024)
    mdp.close_connections()



def train_model():
    model = ModelOperator()
    model.train(num_workers = 0,num_epochs=32,save_model=True)


def full_data_to_ml():
    # cowsay.cow(f"Converting pgn file to sqlite db")    
    # get_data(pgn_file=pgn_file)

    # cowsay.cow(f"Preprocessing data and making rollup table")  
    # preprocess_data()

    # cowsay.cow(f"populating mongodb")    
    # process_data()

    cowsay.cow(f"Training model")    
    train_model()

    cowsay.cow(f"Testing model")    
    board = chess.Board()
    use_model(board=board)
    


def use_model(board: chess.Board = chess.Board()):
    
    move = ba.use_model(board=board)
    # white = 0, black = 1, stalemate = 2
    return move







def test_endgame(board:chess.Board):

    ep = endgamePicker()
    results = ep.find_endgame_best_move(board=board)
    print(results)
    return results



def get_sample_board():
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Bc4")
    board.push_san("Nc6")
    board.push_san("Qh5")
    board.push_san("Nf6")
    #board = chess.Board()

    return board





if __name__ == "__main__":
    main()


