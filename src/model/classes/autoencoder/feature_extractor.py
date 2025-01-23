from chess_engine.src.model.classes.sqlite.models import GamePositionRollup
import numpy as np

from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import  Bitboard_Creator

import chess

class Feature_Extractor(Bitboard_Creator):



    def get_castling_rights_from_board(self,board: chess.Board):
        castling_rights = {'White Kingside Castling': 0,
                    'White Queenside Castling': 0,
                    'Black Queenside Castling': 0,
                    'Black Kingside Castling': 0}
            # Map the castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            castling_rights['White Kingside Castling'] = 1  # White Kingside (K)
        if board.has_queenside_castling_rights(chess.WHITE):
            castling_rights['White Queenside Castling'] = 1  # White Queenside (Q)
        if board.has_kingside_castling_rights(chess.BLACK):
            castling_rights['Black Queenside Castling'] = 1  # Black Kingside (k)
        if board.has_queenside_castling_rights(chess.BLACK):
            castling_rights['Black Kingside Castling'] = 1  # Black Queenside (q)
        return castling_rights

    def get_castling_rights_from_gpr(self,gpr: GamePositionRollup):
        castling_rights = {'White Kingside Castling': 0,
                    'White Queenside Castling': 0,
                    'Black Queenside Castling': 0,
                    'Black Kingside Castling': 0}
            # Map the castling rights
        
        if 'K' in gpr.castling_rights:
            castling_rights['White Kingside Castling'] = 1  # White Kingside (K)
        if 'Q' in gpr.castling_rights:
            castling_rights['White Queenside Castling'] = 1  # White Queenside (Q)
        if 'q' in gpr.castling_rights:
            castling_rights['Black Queenside Castling'] = 1  # Black Kingside (k)
        if 'k' in gpr.castling_rights:
            castling_rights['Black Kingside Castling'] = 1  # Black Queenside (q)

        return castling_rights
    
    def get_metadata_from_gpr(self,gpr: GamePositionRollup):
        metadata = {}
        cr = self.get_castling_rights_from_gpr(gpr=gpr)
        metadata.update(cr)

        return metadata
    
    def get_metadata_from_board(self,board: chess.Board):
        metadata = {}
        cr = self.get_castling_rights_from_board(board=board)
        metadata.update(cr)

        return metadata

    def extract_features_from_board(self,board: chess.Board):
        metadata = self.get_metadata_from_board(board)
        bitboards = self.get_all_bitboards(board)
        return bitboards, metadata

fe = Feature_Extractor()

def get_metadata_from_gpr(gpr: GamePositionRollup):
    castling_rights = fe.get_metadata_from_gpr(gpr)

    return np.array(list(castling_rights.values()))

sample_metada = fe.get_metadata_from_board(board=chess.Board())