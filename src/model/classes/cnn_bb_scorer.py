import chess
from chess_engine.src.model.classes.endgame import endgamePicker
import numpy as np
from math import ceil
from chess_engine.src.model.classes.sqlite.dependencies import board_to_GamePostition
from chess_engine.src.model.classes.sqlite.models import GamePositions
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.metadata_scorer import metaDataBoardEval
import torch
class boardCnnEval:
    def __init__(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = chess.Board()):
        self.half_move_amount = Settings().halfMoveBin

        
        self.ep = endgamePicker()
        
        self.endgameAmount = 5

        self.zeros_matrix = np.zeros((8,8),dtype=int)
        self.black_pieces = [chess.Piece.from_symbol('n'),chess.Piece.from_symbol('b'),chess.Piece.from_symbol('r'),chess.Piece.from_symbol('q'),chess.Piece.from_symbol('k'),chess.Piece.from_symbol('p')]
        self.white_pieces = [chess.Piece.from_symbol('N'),chess.Piece.from_symbol('B'),chess.Piece.from_symbol('R'),chess.Piece.from_symbol('Q'),chess.Piece.from_symbol('K'),chess.Piece.from_symbol('P')]
        self.all_pieces = self.white_pieces + self.black_pieces



    def setup_parameters_board(self,board: chess.Board = None):
        if board is not None:
            self.game = board_to_GamePostition(board=board)
            self.setup_parameters_gamepositions(game=self.game)
            return 1


    def setup_parameters_gamepositions(self,game: GamePositions):
        self.game = game  
        #have to reconstruct full fen instead of just piece_positions
        if game.greater_than_n_half_moves == 1:
            half_moves = self.half_move_amount
            full_moves = 2 * half_moves
        else:
            half_moves = 0
            full_moves = half_moves

        fen = f"{game.piece_positions} {game.turn} {game.castling_rights} {game.en_passant} {half_moves} {full_moves}"
        
        self.board = chess.Board(fen)

        return 0


    def get_features(self):
        scores = self.get_board_scores()
        features = ["moves(id)"] + list(scores.keys())
        return features

    def get_metadata(self):
        metaDataEvaluator = metaDataBoardEval(game=self.game)
        
        dict_results = metaDataEvaluator.get_board_scores()

        turn = self.game.turn

        white_turn = 1 if turn == 'w' else 0
        black_turn = 1 if turn == 'b' else 0

        dict_results["white turn"] = white_turn
        dict_results["black turn"] = black_turn


 
        return dict_results
    
    def en_passant_board(self):
        zeros = self.zeros_matrix.copy()
        if self.game.en_passant != '-':
        # There is a potential en passant target
            target_square = chess.parse_square(self.game.en_passant)
            row, col = divmod(target_square, 8)
            zeros[row, col] = 1
        return zeros


    def is_endgame(self):

        count = self.ep.count_pieces(board=self.board)

        if count <= self.endgameAmount:
            return 1
        else:
            return 0

    def endgame_status(self):
        w_or_b = [0,0]

        results = self.ep.endgame_status(board=self.board)
        if results > 0:
            if self.board.turn:
                w_or_b[0] = 1
            else:
                w_or_b[1] = 1

        elif results < 0:
            if self.board.turn:
                w_or_b[1] = 1
            else:
                w_or_b[0] = 1

        return w_or_b
    def open_tables(self):
        self.ep.open_tables()

    def close_tables(self):
        self.ep.close_tables()

    def get_game_results(self):
        results = []
        dict_results = {}
        means = self.game.win_buckets
        results += means
        dict_results["white mean"] = means[0]
        dict_results["black mean"] = means[1]
        dict_results["stalemate mean"] = means[2]
        return dict_results


    
    
    def get_board_scores(self,board: chess.Board):
        self.setup_parameters_board(board = board)
        dict_results = {}

        dict_results["metadata"] = list(self.get_metadata().values())
        dict_results["positions_data"] = board_to_bitboards(self.board)
        dict_results["game_results"] = list(self.get_game_results().values())

        return dict_results    
    
    def get_game_scores(self,game):
        self.setup_parameters_gamepositions(game=game)
        dict_results = {}

        dict_results["metadata"] = list(self.get_metadata().values())
        dict_results["positions_data"] = board_to_bitboards(self.board)
        dict_results["game_results"] = list(self.get_game_results().values())

        return dict_results    
    

    def get_board_scores_with_labels(self,board: chess.Board):
        self.setup_parameters_board(board = board)
        dict_results = {}

        dict_results["metadata"] = self.get_metadata()
        dict_results["positions_data"] = board_to_bitboards(self.board)   
        dict_results["game_results"] = self.get_game_results()

        return dict_results   
    
    
    def get_board_scores_applied(self,board: chess.Board):
        self.setup_parameters_board(board = board)


        metadata = list(self.get_metadata().values())
        positions_data = board_to_numpy_arrays(self.board)
        

        # return (torch.tensor(positions_data, dtype=torch.float32),
        #         torch.tensor(metadata, dtype=torch.float32))
        return positions_data,metadata
  


def board_to_bitboards(board):
    bitboards = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            bitboard = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == color:
                    bitboard |= 1 << square
            # color_name = 'White' if color == chess.WHITE else 'Black'
            # piece_name = chess.piece_name(piece_type).capitalize()
            bitboards.append(bitboard)
    ep_board = en_passant_bitboard(board=board)
    bitboards.append(ep_board)
    return bitboards

def board_to_numpy_arrays(board):

    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
    numpy_arrays = np.zeros((13, 8, 8), dtype=int)

    for color in (chess.WHITE, chess.BLACK):
        offset = 0 if color == chess.WHITE else 6
        for piece_index, piece_type in enumerate(piece_types):
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.piece_type == piece_type and piece.color == color:
                    row, col = divmod(square, 8)
                    numpy_arrays[offset + piece_index, row, col] = 1
    
    ep_board = en_passant_np_array(board=board)
    numpy_arrays[12,:,:] = ep_board

    return numpy_arrays


def en_passant_bitboard(board):

    en_passant_square = board.ep_square
    

    bitboard = 0

    if en_passant_square is not None:
        bitboard |= chess.BB_SQUARES[en_passant_square]
    
    return bitboard

def en_passant_np_array(board):
    en_passant_square = board.ep_square

    # Initialize an 8x8 numpy array of zeros
    bitboard_array = np.zeros((8, 8), dtype=np.uint8)

    if en_passant_square is not None:
        # Convert the square index to 2D board coordinates
        row = en_passant_square // 8
        col = en_passant_square % 8
        bitboard_array[row, col] = 1

    return bitboard_array

def calc_shapes(batch_size: int = 1024):
    board = chess.Board()
    evaluator = boardCnnEval()        
    data = evaluator.get_board_scores(board=board)
    md_shape = (batch_size,1,len(data['metadata']))
    bb_shape = (batch_size,len(data['positions_data']),8,8)
    gr_shape = (batch_size,len(data['game_results']))

    return (bb_shape,md_shape,gr_shape)