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
from chess_engine.src.model.classes.torch_model import model_operator
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




# mp = cnn_move_picker(neuralNet=nn)


mdp = mongo_data_pipe()

ba = board_analyzer()

mp = move_picker()



def main():
    # test_speeds()
    # pgn_to_db()
    initialize_collections()
    return 0



def create_rollup_table():
    delete_all_rollup_game_positions()
    create_rollup_table(yield_size=500,db=SessionLocal())


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)


def evaluate_mcts_plateau(board: chess.Board):
    set_seeds(10)
    iteration_amts = [64, 128, 258, 512, 1012, 2048, 4096, 8192]
    ucb_constants = [0.1,0.5,1,sqrt(2), 2]
    scorings = [
        [1.5, -1, 5, 0],
        [1, -1, 0.5],
        [1, -1, 0]
    ]
    moves = []
    # Open the CSV file for writing
    if os.path.exists(s.evalModeFile):
        os.remove(s.evalModeFile)
    
    with open(s.evalModeFile, 'w', newline='') as csvfile:

        csv_writer = csv.writer(csvfile)
        # Write the header row
        csv_writer.writerow(['Iteration', 'UCB Constant', 'Scoring', 'Move'])

        for scores in scorings:
            for u in ucb_constants:
                for i in iteration_amts:

                    move = mp.get_best_move(board=board.copy(), 
                                            iterations=i,
                                            ucb_constant=u,
                                            scores=scores)
                    moves.append(move)
                    # Write the data to the CSV file
                    csv_writer.writerow([i, u, scores, move])
                    print(f"iter: {i}, ucb: {u}, scores: {scores}, move: {move}")
    
    print(moves)
    print(board)


def verify_functionality_on_sample_dataset():
    cowsay.cow(f"Converting pgn file to sqlite db")    
    pgn_to_db(pgn_file=pgn_file)
    cowsay.cow(f"populating mongodb")    
    initialize_collections()
    cowsay.cow(f"testing model functions")    
    test_pt_model()
    board = chess.Board()
    use_model(board=board)
    


def use_model(board: chess.Board = chess.Board()):
    
    move = ba.use_model(board=board)
    # white = 0, black = 1, stalemate = 2
    return move

def initialize_collections():
    mdp.open_connections()
    
    mdp.initialize_data(batch_size=512)
    mdp.close_connections()


def eval_board(board: chess.Board):
    return ba.use_model(board=board)

def test_pt_model():
    # initialize_collections()
    model = model_operator()
    model.Create_and_Train_Model(num_workers = 0,num_epochs=16,save_model=False)
    # model.load_and_evaluate_model(model_path=s.torch_model_file)


def pgn_to_db(pgn_file = pgn_file,db: Session = SessionLocal()):

    delete_all_game_positions(db = db)
    pgn_obj = pgn_processor(pgn_file=pgn_file)
    pgn_obj.pgn_fen_to_sqlite()

    
    return 0 


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



def create_csv():
    # Check if the file exists and remove it
    if os.path.exists(eval_file):
        os.remove(eval_file)

    # Create a new CSV file with the column headers
    with open(eval_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        columns = ["epochs","loss","accuracy"]
        writer.writerow(columns)




if __name__ == "__main__":
    set_seeds(10)
    main()


