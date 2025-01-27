import chess.pgn
from tqdm import tqdm

import os
import multiprocessing
from chess_engine.src.model.classes.sqlite.database import SessionLocal
from chess_engine.src.model.classes.sqlite.dependencies import insert_bulk_boards_into_db


class PGNProcessor:
    def __init__(self, pgn_dir, batch_size=5000, num_workers=4):
        """
        :param pgn_dir: Directory containing PGN files.
        :param batch_size: Number of board positions to insert per batch.
        :param num_workers: Number of parallel processes to run.
        """
        self.pgn_dir = pgn_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_all_pgns_parallel(self):
        """ Parallel processing of multiple PGN files using multiprocessing. """
        pgn_files = [os.path.join(self.pgn_dir, f) for f in os.listdir(self.pgn_dir)]

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            pool.map(self.process_single_pgn, pgn_files)

    def process_single_pgn(self, file_path):
        """ Process a single PGN file and insert data in batches. """
        db = SessionLocal()
        board_victors = []

        with open(file_path, encoding='ISO-8859-1') as pgn:
            with tqdm(desc=f"Processing {os.path.basename(file_path)}", unit=" game") as pbar:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break  # End of file
                    if game.headers.get("Result") == '*':
                        continue  # Skip unfinished games

                    victor = self.get_victor(game.headers["Result"])
                    board_victors.extend(self.process_game(game, victor))

                    pbar.update(1)

                    if len(board_victors) >= self.batch_size:
                        insert_bulk_boards_into_db(board_victors, db)
                        board_victors.clear()  # Free memory

        if board_victors:
            insert_bulk_boards_into_db(board_victors, db)  # Insert remaining data
        db.close()

    def process_game(self, game, victor):
        """ Extracts board positions from a game and assigns the winner. """
        board = game.board()
        board_victors = []

        for move in game.mainline_moves():
            board.push(move)

            if not board.turn:  # If it's black's turn, store mirrored position
                append_board = reverse_board(board)
                append_victor = self.flip_victor(victor)
            else:
                append_board = board
                append_victor = victor

            board_victors.append((append_board.copy(), append_victor, board.copy(), victor))

        return board_victors
        

    @staticmethod
    def get_victor(result):
        """ Converts PGN result notation to single-letter victor representation. """
        if result == '1-0':
            return 'w'
        elif result == '0-1':
            return 'b'
        elif result == '1/2-1/2':
            return 's'
        else:
            raise ValueError(f"Unexpected result format: {result}")

    @staticmethod
    def flip_victor(victor):
        """ Swaps white and black victors for mirrored boards. """
        return {'w': 'b', 'b': 'w', 's': 's'}.get(victor, 'NA')
    
    def split_large_pgn_files(self, max_size_mb=50, games_per_file=40000, delete_after_split=False):
        """
        Scans a directory for PGN files and splits any PGN files exceeding max_size_mb into smaller chunks.
        
        :param directory: Directory containing PGN files.
        :param max_size_mb: Maximum allowed file size before splitting (in MB).
        :param games_per_file: Number of games per split PGN file.
        :param delete_after_split: If True, deletes the original PGN file after splitting.
        """
        if not os.path.exists(self.pgn_dir):
            print(f"Error: Directory '{self.pgn_dir}' does not exist.")
            return

        for filename in os.listdir(self.pgn_dir):
            if filename.endswith(".pgn"):
                file_path = os.path.join(self.pgn_dir, filename)
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB

                if file_size_mb > max_size_mb:
                    print(f"Splitting {filename} ({file_size_mb:.2f} MB)...")
                    if self.split_pgn_file(file_path, games_per_file):
                        if delete_after_split:
                            os.remove(file_path)  # Delete the original large PGN file
                            print(f"🗑️ Deleted original PGN file: {filename}")
                else:
                    print(f"Skipping {filename} ({file_size_mb:.2f} MB) - Below size limit.")

    def split_pgn_file(self,input_pgn, games_per_file):
        """
        Splits a large PGN file into smaller PGN chunks.

        :param input_pgn: Path to the large PGN file.
        :param games_per_file: Number of games per smaller PGN file.
        :return: True if splitting was successful, False otherwise.
        """
        output_dir = os.path.dirname(input_pgn)
        base_name = os.path.splitext(os.path.basename(input_pgn))[0]  # Remove .pgn extension

        try:
            with open(input_pgn, encoding='ISO-8859-1') as pgn:
                file_count = 1
                game_count = 0
                output_pgn_path = os.path.join(output_dir, f"{base_name}_chunk_{file_count}.pgn")
                output_pgn = open(output_pgn_path, "w", encoding='ISO-8859-1')

                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break  # End of file

                    print(game, file=output_pgn, end="\n\n")  # Write game to file
                    game_count += 1

                    if game_count >= games_per_file:
                        output_pgn.close()
                        file_count += 1
                        game_count = 0
                        output_pgn_path = os.path.join(output_dir, f"{base_name}_chunk_{file_count}.pgn")
                        output_pgn = open(output_pgn_path, "w", encoding='ISO-8859-1')

                output_pgn.close()
                print(f"✅ {input_pgn} split into {file_count} smaller files.")
                return True  # Splitting successful

        except Exception as e:
            print(f"❌ Error splitting {input_pgn}: {e}")
            return False  # Splitting failed





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