import chess
from chess_engine.src.model.classes.endgame import endgamePicker
import numpy as np
from math import ceil
from chess_engine.src.model.classes.sqlite.dependencies import board_to_GamePostition
from chess_engine.src.model.classes.sqlite.models import GamePositions
from chess_engine.src.model.config.config import Settings
from chess_engine.src.model.classes.metadata_scorer import metaDataBoardEval


class boardCnnEval:
    def __init__(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.half_move_amount = Settings().halfMoveBin
        self.setup_parameters(fen=fen,board=board)
        
        self.ep = endgamePicker()
        
        self.endgameAmount = 5

        self.zeros_matrix = np.zeros((8,8),dtype=int)
        self.black_pieces = [chess.Piece.from_symbol('n'),chess.Piece.from_symbol('b'),chess.Piece.from_symbol('r'),chess.Piece.from_symbol('q'),chess.Piece.from_symbol('k'),chess.Piece.from_symbol('p')]
        self.white_pieces = [chess.Piece.from_symbol('N'),chess.Piece.from_symbol('B'),chess.Piece.from_symbol('R'),chess.Piece.from_symbol('Q'),chess.Piece.from_symbol('K'),chess.Piece.from_symbol('P')]
        self.all_pieces = self.white_pieces + self.black_pieces


    def setup_parameters(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.fen = fen

        self.board = chess.Board(fen=fen)
        if board is not None:
            self.board = board
            self.fen = board.fen()
        self.fen_components = fen.split(" ") 
        return 0


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

    def get_board_scores(self):
        dict_results = {}

        metadata = self.get_metadata()



        positions_data = self.get_positions()

        dict_results.update(metadata)



        dict_results.update(positions_data)
        
        game_results = self.get_game_results()

        dict_results.update(game_results)
        return dict_results
    
    
    def get_board_scores_records(self):
        dict_results = {}

        dict_results["metadata"] = self.get_metadata()



        dict_results["positions_data"] = self.get_positions()


        
        dict_results["game_results"] = self.get_game_results()

        return dict_results    


    def get_positions(self):
        dict_results = {}

        white_attacks,black_attacks = self.w_b_attacks()

        white_advantage = np.where(white_attacks > black_attacks, 1, 0)
        black_advantage = np.where(black_attacks > white_attacks, 1, 0)

        dict_results["white advantage positions"] = white_advantage.flatten()
        dict_results["black advantage positions"] = black_advantage.flatten()

        dict_results["castling positions"] = self.castling_abilities().flatten()
        dict_results["en passant positions"] = self.en_passant_board().flatten()
        
        checks = self.king_check(original_board=self.board)
        dict_results.update(checks)
                
        piece_locations = self.get_all_piece_locations(original_board=self.board)
        dict_results.update(piece_locations)

        attack_paths = self.get_attack_paths(original_board=self.board)
        dict_results.update(attack_paths)

        defended_pieces = self.defended_pieces(original_board=self.board)
        dict_results.update(defended_pieces)

        return dict_results

    def king_check(self,original_board: chess.Board):
        white_results = self.zeros_matrix.copy()
        black_results = self.zeros_matrix.copy()
        dict_results = {}
        board = original_board.copy()

        board.turn = True
        if board.is_check():
            king = board.king(True)
            row,col = divmod(king,8)
            white_results[row,col] = 1

        board.turn = False
        if board.is_check():
            king = board.king(False)
            row,col = divmod(king,8)
            white_results[row,col] = 1

        dict_results["white king check positions"] = white_results.flatten()
        dict_results["black king check positions"] = black_results.flatten()
        return dict_results


    def defended_pieces(self,original_board: chess.Board):
        board = original_board.copy()
        dict_results = {}

        piece_map = board.piece_map()

        white_results = self.defended_pieces_to_matrices(white=True,pieces=self.white_pieces,piece_map=piece_map,board=board)

        black_results = self.defended_pieces_to_matrices(white=False,pieces=self.black_pieces,piece_map=piece_map,board=board)

        dict_results["white knight defended positions"] = white_results[0].flatten()
        dict_results["white bishop defended positions"] = white_results[1].flatten()
        dict_results["white rook defended positions"] = white_results[2].flatten()
        dict_results["white queen defended positions"] = white_results[3].flatten()
        dict_results["white king defended positions"] = white_results[4].flatten()
        dict_results["white pawn defended positions"] = white_results[5].flatten()

        dict_results["black knight defended positions"] = black_results[0].flatten()
        dict_results["black bishop defended positions"] = black_results[1].flatten()
        dict_results["black rook defended positions"] = black_results[2].flatten()
        dict_results["black queen defended positions"] = black_results[3].flatten()
        dict_results["black king defended positions"] = black_results[4].flatten()
        dict_results["black pawn defended positions"] = black_results[5].flatten()
        
        white_pieces = white_results[0]
        for i in range(1,len(white_results)):
            white_pieces += white_results[i]

        black_pieces = black_results[0]
        for i in range(1,len(black_results)):
            black_pieces += black_results[i]

        total_pieces = black_pieces + white_pieces

        dict_results["all defended positions"] = total_pieces.flatten()

        return dict_results

    def defended_pieces_to_matrices(self, pieces, piece_map, white: bool, board: chess.Board):
        zeros = self.zeros_matrix.copy()
        results = []
        for _ in range(6):
            results.append(np.copy(zeros))

        for position, piece in piece_map.items():
            if piece in pieces:
                result = pieces.index(piece)

                # Get the number of attackers on this square
                attackers = board.attackers(not white, position) # Using `not white` to get attackers of the opposite color
                defenders = board.attackers(white, position)
                defended = len(defenders) - len(attackers)
                
                if defended > 0:
                    row, col = divmod(position, 8)
                    results[result][row, col] = 1

        return results



    def attacks_to_matrix(self,pieces, moves,piece_map):
        zeros = self.zeros_matrix.copy()
        results = []
        for _ in range(6):
            results.append(np.copy(zeros))


        for move in moves:
            from_square = move.from_square
            to_square = move.to_square
            piece = piece_map[from_square]
            for i in range(0,len(pieces)):
                current_piece = pieces[i]
                if piece == current_piece:
                    row,col = divmod(to_square,8)
                    results[i][row,col] = 1
                    

        return results

    def get_attack_paths(self,original_board: chess.Board):
        board = original_board.copy()
        dict_results = {}

        piece_map = board.piece_map()

        board.turn = True
        moves = board.legal_moves
        white_results = self.attacks_to_matrix(pieces=self.white_pieces,moves=moves,piece_map=piece_map)
        
        board.turn = False
        moves = board.legal_moves
        black_results = self.attacks_to_matrix(pieces=self.black_pieces,moves=moves,piece_map=piece_map)

        dict_results["white knight attack positions"] = white_results[0].flatten()
        dict_results["white bishop attack positions"] = white_results[1].flatten()
        dict_results["white rook attack positions"] = white_results[2].flatten()
        dict_results["white queen attack positions"] = white_results[3].flatten()
        dict_results["white king attack positions"] = white_results[4].flatten()
        dict_results["white pawn attack positions"] = white_results[5].flatten()

        dict_results["black knight attack positions"] = black_results[0].flatten()
        dict_results["black bishop attack positions"] = black_results[1].flatten()
        dict_results["black rook attack positions"] = black_results[2].flatten()
        dict_results["black queen attack positions"] = black_results[3].flatten()
        dict_results["black king attack positions"] = black_results[4].flatten()
        dict_results["black pawn attack positions"] = black_results[5].flatten()
        
        white_pieces = white_results[0]
        for i in range(1,len(white_results)):
            white_pieces += white_results[i]

        black_pieces = black_results[0]
        for i in range(1,len(black_results)):
            black_pieces += black_results[i]


        white_minus_black = white_pieces - black_pieces

        black_pieces_attacks = np.where(black_pieces > 0, 1, 0)
        white_pieces_attacks = np.where(white_pieces > 0, 1, 0)

        dict_results["white attack positions"] = white_pieces_attacks.flatten()
        dict_results["black attack positions"] = black_pieces_attacks.flatten()

        white_advantage = np.where(white_minus_black > 0, 1, 0)
        black_advantage = np.where(white_minus_black < 0, 1, 0)

        dict_results["white advantage attack positions"] = white_advantage.flatten()
        dict_results["black advantage attack positions"] = black_advantage.flatten()

        return dict_results


    def piece_positons(self,white: bool,piece: str):
        if white:
            piece = piece.upper()
        else:
            piece = piece.lower()
        
    def get_all_piece_locations(self,original_board: chess.Board):
        board = original_board.copy()
        dict_results = {}

        piece_map = board.piece_map()
        white_results = self.pieces_to_matrix(pieces=self.white_pieces,piece_map=piece_map)
        black_results = self.pieces_to_matrix(pieces=self.black_pieces,piece_map=piece_map)

        dict_results["white knight positions"] = white_results[0].flatten()
        dict_results["white bishop positions"] = white_results[1].flatten()
        dict_results["white rook positions"] = white_results[2].flatten()
        dict_results["white queen positions"] = white_results[3].flatten()
        dict_results["white king positions"] = white_results[4].flatten()
        dict_results["white pawn positions"] = white_results[5].flatten()

        dict_results["black knight positions"] = black_results[0].flatten()
        dict_results["black bishop positions"] = black_results[1].flatten()
        dict_results["black rook positions"] = black_results[2].flatten()
        dict_results["black queen positions"] = black_results[3].flatten()
        dict_results["black king positions"] = black_results[4].flatten()
        dict_results["black pawn positions"] = black_results[5].flatten()
        
        white_pieces = white_results[0]
        for i in range(1,len(white_results)):
            white_pieces += white_results[i]

        black_pieces = black_results[0]
        for i in range(1,len(black_results)):
            black_pieces += black_results[i]

        total_pieces = white_pieces + black_pieces

        black_pieces = np.where(black_pieces > 0, 1, 0)
        white_pieces = np.where(white_pieces > 0, 1, 0)

        dict_results["white positions"] = white_pieces.flatten()
        dict_results["black positions"] = black_pieces.flatten()
        dict_results["white black positions"] = total_pieces.flatten()

        return dict_results

    def all_team_piece_locations(self):
        #white = 1, black = -1
        return 0    
        

    
    def pieces_to_matrix(self,pieces, piece_map):
        zeros = self.zeros_matrix.copy()
        results = []
        for _ in range(6):
            results.append(np.copy(zeros))
        for i in range(0,len(pieces)):
            current_piece = pieces[i]
            current_results = results[i]
            keys = list(piece_map.keys())

            for key in keys:
                if piece_map[key] == current_piece:
                    row,col = divmod(key,8)
                    current_results[row][col] += 1
                    piece_map.pop(key)
            current_results = current_results / 2
        return results
    
    def castling_abilities(self):
        zeros = self.zeros_matrix.copy()

        for rights in self.game.castling_rights:
            if rights == 'q':
                zeros[0][2] = 1
            elif rights == 'Q':
                zeros[7][2] = 1
            elif rights == 'k':
                zeros[0][6] = 1                
            elif rights == 'K':
                zeros[7][6] = 1                

        return zeros

    def w_b_attacks(self):
        white_attacks = self.calculate_square_attacks(white=True)
        black_attacks = self.calculate_square_attacks(white=False)
        return white_attacks, black_attacks
    
    def calculate_square_attacks(self,white:bool):
        attack_matrix = self.zeros_matrix.copy()
        pro = chess.WHITE if white else chess.BLACK
        # Iterate over all squares
        for square in chess.SQUARES:
            attackers = self.board.attackers(pro, square)
            count_attackers = str(attackers).count("1")

            # Convert the square index to row and column for the matrix
            row, col = divmod(square, 8)
            attack_matrix[row, col] = count_attackers

        return attack_matrix

    def get_features(self):
        board = chess.Board()
        game = board_to_GamePostition(board=board,victor='w')
        self.setup_parameters_gamepositions(game=game)
        scores = self.get_board_scores()
        features =  list(scores.keys())
        return features
    
def sequence_to_rc(seq: int):
        row = 7 - (int(ceil((seq + 1)/8)) - 1) 
        col = (seq+1) % 8 - 1
        return row,col
