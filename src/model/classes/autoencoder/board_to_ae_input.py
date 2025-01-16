from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
import numpy as np
from tqdm import tqdm
from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict, Bitboard_Creator

from chess_engine.src.model.classes.sqlite.database import  get_db
from chess_engine.src.model.config.config import data_settings
import chess

class Feature_Extractor(Bitboard_Creator):
    def get_metadata_from_board(self,board: chess.Board):
        metadata = {'White Kingside Castling': 0,
                    'White Queenside Castling': 0,
                    'Black Queenside Castling': 0,
                    'Black Kingside Castling': 0}
            # Map the castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            metadata['White Kingside Castling'] = 1  # White Kingside (K)
        if board.has_queenside_castling_rights(chess.WHITE):
            metadata['White Queenside Castling'] = 1  # White Queenside (Q)
        if board.has_kingside_castling_rights(chess.BLACK):
            metadata['Black Queenside Castling'] = 1  # Black Kingside (k)
        if board.has_queenside_castling_rights(chess.BLACK):
            metadata['Black Kingside Castling'] = 1  # Black Queenside (q)
        return metadata
    
    def extract_features_from_board(self,board: chess.Board):
        metadata = self.get_metadata_from_board(board)
        bitboards = self.get_all_bitboards(board)
        return bitboards, metadata