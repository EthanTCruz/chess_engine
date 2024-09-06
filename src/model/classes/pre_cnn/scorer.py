import chess
from chess_engine.src.model.classes.endgame import endgamePicker

class boardEval:
    def __init__(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.setup_parameters(fen=fen,board=board)
        
        self.ep = endgamePicker()
        
        self.endgameAmount = 5

    def setup_parameters(self,fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',board: chess.Board = None):
        self.fen = fen

        self.board = chess.Board(fen=fen)
        if board is not None:
            self.board = board
            self.fen = board.fen()
        self.fen_components = fen.split(" ") 




    def get_features(self):
        scores = self.get_board_scores()
        features = ["moves(id)"] + list(scores.keys())
        return features

    def get_board_scores(self,victor="NA"):

        
        dict_results = {}
        
        white_amounts, black_amounts = self.get_piece_amounts()
        dict_results["white pawns"] = white_amounts[0]
        dict_results["white knights"] = white_amounts[1]
        dict_results["white bishops"] = white_amounts[2]
        dict_results["white rooks"] = white_amounts[3]
        dict_results["white queens"] = white_amounts[4]
        dict_results["black pawns"] = black_amounts[0]
        dict_results["black knights"] = black_amounts[1]
        dict_results["black bishops"] = black_amounts[2]
        dict_results["black rooks"] = black_amounts[3]
        dict_results["black queens"] = black_amounts[4]
        dict_results["total black pieces"] = sum(black_amounts)
        dict_results["total white pieces"] = sum(white_amounts)



        
        

        piece_pairs = self.knight_bishop_pairs()
        dict_results["white has bishop pair"] = piece_pairs["white has bishop pair"]
        dict_results["black has bishop pair"] = piece_pairs["black has bishop pair"]
        dict_results["white has knight bishop pair"] = piece_pairs["white has knight bishop pair"]
        dict_results["black has knight bishop pair"] = piece_pairs["black has knight bishop pair"]                
        dict_results["white has knight pair"] = piece_pairs["white has knight pair"]
        dict_results["black has knight pair"] = piece_pairs["black has knight pair"]

        moves = self.number_of_moves()
        dict_results["white moves"] = moves["white"]
        dict_results["black moves"] = moves["black"]

        if moves["black"] == 0:
            dict_results["white to black moves"] = 0
        else:
            dict_results["white to black moves"] = (moves["white"]/(moves["black"]))

        dict_results["white knight moves"] = moves["white N"]
        dict_results["white bishop moves"] = moves["white B"]
        dict_results["white rook moves"] = moves["white R"]
        dict_results["white queen moves"] = moves["white Q"]

        dict_results["black knight moves"] = moves["black N"]
        dict_results["black bishop moves"] = moves["black B"]
        dict_results["black rook moves"] = moves["black R"]
        dict_results["black queen moves"] = moves["black Q"]

        '''
        dict_results["knight ratio"] = (dict_results["knight moves"]+1)/(dict_results["black knight moves"]+1)
        dict_results["bishop ratio"] = (dict_results["bishop moves"]+1)/(dict_results["black bishop moves"]+1)
        dict_results["rook ratio"] = (dict_results["rook moves"]+1)/(dict_results["black rook moves"]+1)
        dict_results["queen ratio"] = (dict_results["queen moves"]+1)/(dict_results["black queen moves"]+1)
        '''

        middle_square_possesion = self.middle_square_attacks()
        dict_results["e4 possesion"] = middle_square_possesion[0]
        dict_results["e5 possesion"] = middle_square_possesion[1]
        dict_results["d4 possesion"] = middle_square_possesion[2]
        dict_results["d5 possesion"] = middle_square_possesion[3]

        middle_square_possesion = self.middle_square_occupation()
        dict_results["e4 occupation"] = middle_square_possesion[0]
        dict_results["e5 occupation"] = middle_square_possesion[1]
        dict_results["d4 occupation"] = middle_square_possesion[2]
        dict_results["d5 occupation"] = middle_square_possesion[3]
        
        time = self.get_game_time()
        dict_results["Beginning Game"] = time[0]
        dict_results["Middle Game"] = time[1]
        dict_results["End Game"] = time[2]

        # turn = self.turn()
        # dict_results["white"] = turn[0]
        # dict_results["black"] = turn[1]

        dict_results["halfmove_clock"] = self.halfmove_clock()
        dict_results["repetition"] = self.repeated_position()
        dict_results["insufficient material"] = self.insufficient_material()
        
        can_be_drawn = self.can_be_drawn(white=False)
        dict_results["black can be drawn"] = can_be_drawn[0]

        dict_results["black promote to queen"] = can_be_drawn[1]
        
        #Increase queen count feature
        
        can_be_drawn = self.can_be_drawn(white=True)
        dict_results["white can be drawn"] = can_be_drawn[0]

        dict_results["white promote to queen"] = can_be_drawn[1]


        if self.board.turn:
            dict_results["can be drawn"] = dict_results["white can be drawn"]
        else:
            dict_results["can be drawn"] = dict_results["black can be drawn"]

        dict_results["black queen can be taken"] = self.is_queen_in_danger(white=False)
        dict_results["white queen can be taken"] = self.is_queen_in_danger(white=True)

        attacks = self.attacks_per_side()
        dict_results["white attacks"] = attacks[0]
        dict_results["black attacks"] = attacks[1]
        black_can_attack = self.can_attack(attacks[1])
        white_can_attack = self.can_attack(attacks[0])
        dict_results["white can attack"] = white_can_attack
        dict_results["black can attack"] = black_can_attack
        
        if dict_results["End Game"] == 1:
            endgame_scores = self.endgame_status()
        else:
            endgame_scores = [0,0]

        dict_results["white wdl"] = endgame_scores[0]
        dict_results["black wdl"] = endgame_scores[1]
        


        win = self.is_checkmate()
        # dict_results["white win"] = win[0]
        # dict_results["black win"] = win[1]
        dict_results["checkmate"] = sum(win)
        dict_results["stalemate"] = self.is_stalemate()


        dict_results["w/b"] = victor

        return dict_results
    
    def open_tables(self):
        self.ep.open_tables()

    def close_tables(self):
        self.ep.close_tables()
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

    def is_queen_in_danger(self,white: bool):


        queens = self.board.pieces(chess.QUEEN, white)


        original_state = self.board.copy()


        original_state.turn = white

        for move in original_state.legal_moves:
            if move.to_square in queens:

                original_state.pop()
                return 1
        return 0

    def can_attack(self,attacks):
        if attacks > 0:
            return 1
        else:
            return 0

    def insufficient_material(self):
        if self.board.is_insufficient_material():
            return 1
        else:
            return 0
        


    def can_be_drawn(self, white:bool):
        temp = self.board.copy()
        temp.turn = white
        can_be_drawn = 0
        queen_promotion = 0
        bishop_promotion = 0
        knight_promotion = 0
        rook_promotion = 0
        for move in self.board.legal_moves:
            if temp.is_pseudo_legal(move) and temp.is_into_check(move):  # Ensure move is legal and doesn't put own king in check
                temp.push(move)
                if temp.result() == "1/2-1/2":
                    can_be_drawn = 1
                    break
                elif move.promotion == chess.QUEEN:  # Check if the move promotes a pawn to a queen
                    queen_promotion = 1
                elif move.promotion == chess.BISHOP:  # Check if the move promotes a pawn to a queen
                    bishop_promotion = 1
                elif move.promotion == chess.KNIGHT:  # Check if the move promotes a pawn to a queen
                    knight_promotion = 1
                elif move.promotion == chess.ROOK:  # Check if the move promotes a pawn to a queen
                    rook_promotion = 1
                    # Add any specific handling for pawn promotion here
                temp.pop()
        return [can_be_drawn,queen_promotion,rook_promotion,bishop_promotion,knight_promotion]


    def attacks_per_side(self):
        temp = self.board.copy()
        temp.turn = True
        white_attacks = self.attacks(board=temp)
        temp.turn = False
        black_attacks = self.attacks(board=temp)
        return [white_attacks,black_attacks]


    def attacks(self,board:chess.Board):
        amount_of_attacks = 0
        moves = [board.san(move) for move in board.legal_moves]
        for move in moves:
            if 'x' in move:
                amount_of_attacks += 1
        return amount_of_attacks

    def repeated_position(self):
        rep = 0
        for i in range(0,5):
            if self.board.is_repetition(i):
                rep += 1
        return rep
    def halfmove_clock(self):
        return self.board.halfmove_clock
    def middle_square_attacks(self):
        middle_squares =[chess.E4,chess.E5,chess.D4,chess.D5]
        square_possesion = []

        for square in middle_squares:
            

            attackers = str(self.board.attackers(chess.WHITE,square=square)).count("1") - str(self.board.attackers(chess.BLACK,square=square)).count("1")
            
            square_possesion.append(attackers)
            #positive means white
        return square_possesion

    def middle_square_occupation(self):
        middle_squares =[chess.E4,chess.E5,chess.D4,chess.D5]
        square_occupation = []

        for square in middle_squares:
            

            occupation = self.get_piece_color(square=square)
            
            square_occupation.append(occupation)
            #positive means white
        return square_occupation
    

    def get_piece_color(self,square):
        

        if not self.board.piece_at(square):
            return 0.5

        
        # Check the color of the piece on the square
        if self.board.color_at(square) == chess.WHITE:
            return 1
        else:
            return 0




    def get_piece_amounts(self):
        white_pieces = ['P','N','B','R','Q']
        black_pieces = ['p','n','b','r','q']
        white_amounts = []
        black_amounts = []

        fen = self.fen_components[0]

        for i in range(0,5):
            white_amounts.append(fen.count(white_pieces[i]))
            black_amounts.append(fen.count(black_pieces[i]))
            


        return white_amounts,black_amounts
    
    def knight_bishop_pairs(self):
        results = {}
        fen = self.fen_components[0]
        
        white_bishop = 'B'
        black_bishop = 'b'
        white_knight  = 'N'
        black_knight = 'n'


        black_bishop_count = fen.count(black_bishop)
        white_bishop_count = fen.count(white_bishop)
        black_knight_count = fen.count(black_knight)
        white_knight_count = fen.count(white_knight)

        results["black knight"] = black_knight_count
        results["white knight"] = white_knight_count
        results["black bishops"] = black_bishop_count
        results["white bishops"] = white_bishop_count
        
        white_bishop_pair = 0
        white_knight_pair = 0
        white_knight_bishop_pair = 0
        black_bishop_pair = 0
        black_knight_pair = 0
        black_knight_bishop_pair = 0
        
        if white_bishop_count >= 2:
            white_bishop_pair = 1
        if white_knight_count >= 2:
            white_knight_pair = 1
        if white_knight_count >= 1 and white_bishop_count >= 1:
            white_knight_bishop_pair = 1

        if black_bishop_count >= 2:
            black_bishop_pair = 1
        if black_knight_count >= 2:
            black_knight_pair = 1
        if black_knight_count >= 1 and black_bishop_count >= 1:
            black_knight_bishop_pair = 1

        results["black has knight pair"] = black_knight_pair
        results["black has knight bishop pair"] = black_knight_bishop_pair
        results["black has bishop pair"] = black_bishop_pair

        results["white has knight pair"] = white_knight_pair
        results["white has knight bishop pair"] = white_knight_bishop_pair
        results["white has bishop pair"] = white_bishop_pair
        return results

    def get_game_time(self):
        #could be split up into two features for each side?
        board = self.fen_components[0]
        count = len(board)
        count = count - board.count("/")
        bgame = 0
        egame = 0
        mgame = 0

        for i in range(1,9):
            count = count - board.count(str(i))

        if (count) >= (30):
            bgame=1
        elif count > self.endgameAmount:
            mgame = 1
        else:
            egame = 1
        return ([bgame,mgame,egame])

    def is_checkmate(self):
        w_or_b = [0,0]
        if self.board.is_game_over():
            results = self.board.result()
            if results == '1-0':
                w_or_b[0] = 1
            elif results == '0-1':
                w_or_b[1] = 1
            
        return w_or_b
        
    def is_stalemate(self):
        if  self.board.is_stalemate():
            return 1
        else:
            return 0

    def turn(self):
        w_or_b = [0,0]
        if self.board.turn:
            w_or_b[0] = 1
        else:
            w_or_b[1] = 1
        return w_or_b

    def possible_moves(self,white):

        temp = self.board.copy()
        temp.turn = white
        #max moves from one side is 187?
        moves = list(temp.legal_moves)
        moves = [temp.san(chess.Move.from_uci(str(move))) for move in moves]
        return (moves)

    def number_of_moves(self):
        black_moves = self.possible_moves(white = False)
        white_moves = self.possible_moves(white = True)
        black = len(black_moves)
        white = len(white_moves)
        data = {}

        pieces = ['P','N','B','R','Q']
        for piece in pieces:
            data[f'white {piece}'] = 0
            data[f'black {piece}'] = 0
            new_moves = []
            for i in range(len(white_moves)):
                if white_moves[i][0] == piece:
                    data[f'white {piece}'] = data[f'white {piece}'] + 1
                else:
                    new_moves.append(white_moves[i])
            white_moves=new_moves
            new_moves = []
            for i in range(len(black_moves)):
                if black_moves[i][0] == piece:
                    data[f'black {piece}'] = data[f'black {piece}'] + 1
                else:
                    new_moves.append(black_moves[i])
            black_moves=new_moves                

        data["white"] = white
        data["black"] = black

        return data
    
    def get_king_pressure(self):
        return 0
    
    def get_opp_king_pressure(self):
        return 0
    
    def get_king_xray(self):
        
        return 0
    
    def get_opp_king_xray(self):
        return 0