import chess.pgn
from tqdm import tqdm
from sqlalchemy.orm import  Session
import os

from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.sqlite.dependencies import insert_bulk_boards_into_db


class pgn_processor():
    def __init__(self,pgn_file) -> None:
        self.pgn_file = pgn_file

    


    def pgn_fen_to_sqlite(self, db: Session = SessionLocal()):
        for filename in os.listdir(self.pgn_file):
            file = f"{self.pgn_file}/{filename}"
            total_games = count_games_in_pgn(pgn_file=file)
            with open(file, encoding='ISO-8859-1') as pgn:  # Specify the encoding
                for _ in tqdm(range(total_games), desc=f"Processing {filename} Games to DB"):
                    game = chess.pgn.read_game(pgn)

                    if game is None:
                        break  # end of file
                    if game.headers["Result"] == '*':
                        continue  # skip unfinished games
                    board = game.board()
                    board_victors = []
                    victor = 'NA'
                    if game.headers["Result"] == '1-0':
                        victor = 'w'
                    elif game.headers["Result"] == '0-1':
                        victor = 'b'
                    elif game.headers["Result"] == '1/2-1/2':
                        victor = 's'
                    else:
                        print(game.headers["Result"])
                        raise Exception("No winner")

                    for move in game.mainline_moves():
                        board.push(move=move)
                        board_victors.append((board.copy(), victor))
                    insert_bulk_boards_into_db(board_victors=board_victors, db=db)



def count_games_in_pgn(pgn_file):
    count = 0
    with open(pgn_file, encoding='ISO-8859-1') as pgn:  # Specify the encoding
        while chess.pgn.read_game(pgn) is not None:
            count += 1
    return count
