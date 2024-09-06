import chess

class ezboard(chess.Board):

    def __init__(self, fen=chess.STARTING_FEN, move_list = []):
        super().__init__(fen)
        self.FEN = fen
        self.move_list = move_list


    def reset(self,fen=chess.STARTING_FEN):
        self.set_fen(fen)
        self.move_list = []

    def reset_board(self):
        self.set_fen(chess.STARTING_FEN)

    def move(self, move_string):
        move = chess.Move.from_uci(move_string)
        if move in self.legal_moves:
            self.push(move)
            self.move_list.append(move_string)
            return True
        else:
            return False

    def go_back(self,num_of_moves = 1):
        if num_of_moves > len(self.move_list)+1:
            raise Exception("Number of moves to regress exceeds size of move list")
        temp = self.move_list[:(-num_of_moves)]
        self.reset(fen=self.FEN)
        for move in temp:
            self.move(move_string=move)

def test_ez_board():
    board = ezboard()
    board.move("e2e4")  # Makes the move e2e4
    print(board.fen())
    board.move("e7e5")  # Makes the move e7e5
    board.move("g1f3")  # Makes the move g1f3
    board.go_back(num_of_moves=1)
    print(board.fen())
    print(board.move_list)
