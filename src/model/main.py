import sys
import os

import cowsay
import chess

from math import log,sqrt,e,inf

import os
if os.path.exists('./chess_engine'):
    os.chdir('./chess_engine')
sys.path.append('../')

from sqlalchemy.orm import  Session
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.sqlite.dependencies import (
    delete_all_game_positions,
    delete_all_rollup_game_positions,
)
from chess_engine.src.model.classes.preprocess_data.dataset_splitter import  create_rollup_table
from chess_engine.src.model.classes.collect_data.pgn_processor import pgn_processor

from chess_engine.src.model.config.config import settings, model_settings

from chess_engine.src.model.classes.applied_model.endgame import endgamePicker
from chess_engine.src.model.classes.model_training.torch_model import ModelOperator, set_seed

# from chess_engine.src.model.classes.npz_piping.create_npz_files import db_to_npz_files
from chess_engine.src.model.classes.process_data.create_h5_files import db_to_hdf5_files
from chess_engine.src.model.classes.dataloader.dataloader import get_dataloader_full_retrieval_time


from chess_engine.src.model.classes.autoencoder.model_operator import AutoencoderTrainer


set_seed()


pgn_file = settings.pgn_file

epochs = model_settings.Epochs


if settings.useSamplePgn:
    pgn_file=settings.samplePgn

# pgn_file=settings.samplePgn



# mp = cnn_move_picker(neuralNet=nn)


# mdp = mongo_data_pipe()

# ba = board_analyzer()

# mp = move_picker()



def main():

    if settings.getData:
        get_data(pgn_file)

    if settings.preprocessData:
        preprocess_data()

    if settings.processData:
        process_data()

    if settings.trainEncoder:
        train_encoder()

    if settings.trainModel:
        train_model()

    return 0




def train_encoder():
    cowsay.cow(f"Training encoder")  
    ae = AutoencoderTrainer()
    ae.train_encoder()


def get_data(pgn_file = pgn_file,db: Session = SessionLocal()):
    cowsay.cow(f"Converting PGN's to SQLITE")    
    delete_all_game_positions(db = db)
    pgn_obj = pgn_processor(pgn_file=pgn_file)
    pgn_obj.pgn_fen_to_sqlite()

def preprocess_data():
    cowsay.cow(f"Converting pgn file to sqlite db")    
    delete_all_rollup_game_positions()
    create_rollup_table(yield_size=256,db=SessionLocal())


def process_data():
    cowsay.cow(f"Splitting dataset into train, validation and test sets")  
    db_to_hdf5_files()


def test_dataloader():
    get_dataloader_full_retrieval_time()


def train_model():
    cowsay.cow(f"Training model")  
    model = ModelOperator()
    model.train(num_epochs=epochs,save_model=True)


def full_data_to_ml():
    cowsay.cow(f"Converting pgn file to sqlite db")    
    # get_data(pgn_file=pgn_file)

    cowsay.cow(f"Preprocessing data and making rollup table")  
    # preprocess_data()

    cowsay.cow(f"populating mongodb")    
    process_data()

    cowsay.cow(f"Training model")    
    train_model()

    # cowsay.cow(f"Testing model")    
    # board = chess.Board()
    # use_model(board=board)
    


def use_model(board: chess.Board = chess.Board()):
    ba = board_analyzer()
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


