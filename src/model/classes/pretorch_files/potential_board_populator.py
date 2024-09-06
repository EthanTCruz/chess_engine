import chess
from chess_engine.src.model.config.config import Settings






class populator():

    def __init__(self,**kwargs):
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        #for holding all moves instead of redis
        self.total_moves = {}
        if "score_depth" not in kwargs:
            self.score_depth=s.score_depth
        else:
            self.score_depth = kwargs["score_depth"]   

        if "board" not in kwargs:
            raise Exception("Need initial baord")
        else:
            self.board = kwargs["board"].copy()



    def new_board(self,board: chess.Board):
        self.board = board.copy()

    def set_depth_and_board(self,depth: int, board: chess.Board):
        self.board = board.copy()
        self.score_depth = depth

    def get_all_moves(self):
        self.total_moves = {}
        self.get_moves(board=self.board,moves=self.get_legal_moves(board=self.board))

    def return_total_moves(self):
        return self.total_moves

    def get_legal_moves(self,board: chess.Board):
        legal_moves =  [move.uci() for move in board.legal_moves]
        return(legal_moves)
        
        

    def get_moves(self,board: chess.Board, moves: list[str] = [],initial_movestack_length: int = 0,first_iter: bool = True):
        move_dict = {}
        if initial_movestack_length == 0 and first_iter:
            initial_movestack_length = len(board.move_stack)

        for move in moves:
            try:
                
                board.push_uci(move)
                value = "u"

                current_depth = (len(board.move_stack) - initial_movestack_length)
                move_list = [move.uci() for move in board.move_stack[(-current_depth):]]

                if board.is_checkmate():
                    value = "w"
                    if board.turn:
                        value = "b"
                    self.total_moves[f'{str(move_list)}:{ board.fen()}'] = [value,board.copy()]

                elif board.result() == "1/2-1/2":
                    value = "s"
                    self.total_moves[f'{str(move_list)}:{ board.fen()}'] = [value,board.copy()]

                elif current_depth >= self.score_depth :
                    self.total_moves[f'{str(move_list)}:{ board.fen()}'] = [value,board.copy()]

                else:
                    legal_moves = self.get_legal_moves(board)
                    if legal_moves:
                        sub_dict = self.get_moves(board,moves=legal_moves,initial_movestack_length=initial_movestack_length,first_iter=False)
                        move_dict.update(sub_dict)

            except ValueError:
                pass
            board.pop()
        return move_dict
    
        