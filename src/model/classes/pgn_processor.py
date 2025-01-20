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

                        if not board.turn:
                            append_board = reverse_board(board=board)

                            if victor == 'b':
                                append_victor = 'w'
                            elif victor == 'w':
                                append_victor = 'b'
                            else:
                                append_victor = 's'

                        else:

                            append_board = board
                            append_victor = victor

                        board_victors.append((append_board.copy(), append_victor,board.copy(),victor))
                    insert_bulk_boards_into_db(board_victors=board_victors, db=db)



def count_games_in_pgn(pgn_file):
    count = 0
    with open(pgn_file, encoding='ISO-8859-1') as pgn:  # Specify the encoding
        while chess.pgn.read_game(pgn) is not None:
            count += 1
    return count

def reverse_board(board: chess.Board) -> chess.Board:
    # Create a new empty board
    new_board = chess.Board(None)  # Empty board

    # Iterate over all squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Swap color
            new_piece = chess.Piece(
                piece.piece_type,
                chess.WHITE if piece.color == chess.BLACK else chess.BLACK
            )
            # Place on flipped position
            flipped_square = chess.square_mirror(square)
            new_board.set_piece_at(flipped_square, new_piece)

    # Set castling rights
    new_castling_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        new_castling_rights |= chess.BB_H8  # White's kingside castling becomes black's kingside castling
    if board.has_queenside_castling_rights(chess.WHITE):
        new_castling_rights |= chess.BB_A8  # White's queenside castling becomes black's queenside castling
    if board.has_kingside_castling_rights(chess.BLACK):
        new_castling_rights |= chess.BB_H1  # Black's kingside castling becomes white's kingside castling
    if board.has_queenside_castling_rights(chess.BLACK):
        new_castling_rights |= chess.BB_A1  # Black's queenside castling becomes white's queenside castling

    new_board.castling_rights = new_castling_rights
    #  KQkq
    # Flip en passant square if it exists
    if board.ep_square is not None:
        new_board.ep_square = chess.square_mirror(board.ep_square)
    else:
        new_board.ep_square = None

    # Set the turn to white
    new_board.turn = chess.WHITE

    return new_board