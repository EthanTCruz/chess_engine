import chess
from multiprocessing import Pool
import csv
import os
import pandas as pd
from sqlalchemy.orm import  Session
from tqdm import tqdm


from chess_engine.src.model.classes.cnn_bb_scorer import boardCnnEval
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.sqlite.dependencies import  fetch_all_game_positions_rollup,get_rollup_row_count,board_to_GamePostition
from chess_engine.src.model.classes.sqlite.models import GamePositions
from chess_engine.src.model.classes.sqlite.database import SessionLocal

metadata_key = 'metadata'
bitboards_key = 'positions_data'
results_key = 'game_results'


class game_analyzer:
    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "output_file" not in kwargs:
            self.output_file=s.scores_file
        else:
            self.output_file = kwargs["output_file"]
        

        if "persist_data" not in kwargs:
            self.persist_data = False
        else:
            self.persist_data = kwargs["persist_data"]

        self.evaluator = boardCnnEval()


        self.recordsDataFile = s.recordsData
        
    def open_endgame_tables(self):
        self.evaluator.open_tables()

    def close_endgame_tables(self):
        self.evaluator.close_tables()
        


    @staticmethod
    def worker(game):
        if game is None: 
            return 1
        # This static method will be the worker function
        # Instantiate game_analyzer and evaluate_board here, since each process will have its own instance
        analyzer = game_analyzer()
        return analyzer.evaluate_board(game=game)





    def evaluate_board(self,game: GamePositions):
        return self.evaluator.get_game_scores(game=game)




    def process_sqlite_boards(self,db: Session = SessionLocal()):
        self.create_csv()

        row_count = get_rollup_row_count(db=db)
        with open(self.output_file, 'a', newline='') as gameEvalfile:
            writer = csv.writer(gameEvalfile)
            batch = fetch_all_game_positions_rollup(yield_size=500, db=db)
            
            # Wrap the generator with tqdm
            for game in tqdm(batch, total=row_count, desc="Processing Feature Data"):
                try:
                    if game is None:
                        return 1

                    scores = self.evaluate_board(game=game)
                    row = list(scores.values())
                    writer.writerow(row)

                except Exception as e:
                    raise Exception(e)


        

    




