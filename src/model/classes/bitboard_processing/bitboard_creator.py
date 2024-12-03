import chess
import numpy as np




class Bitboard_Creator:
    def __init__(self):
        self.color_map = {chess.WHITE: "white", chess.BLACK: "black"}
        self.piece_map = {
                        chess.PAWN: "pawn",
                        chess.KNIGHT: "knight",
                        chess.BISHOP: "bishop",
                        chess.ROOK: "rook",
                        chess.QUEEN: "queen",
                        chess.KING: "king"
                        }
        
    def get_base_bitboards_dict(self,board):
        bitboards = {
            f"{self.color_map[color]} {self.piece_map[piece]}": str(int(board.pieces(piece, color)))
            for color in [chess.WHITE, chess.BLACK]
            for piece in chess.PIECE_TYPES
        }
        return bitboards
    def get_all_bitboards(self,board: chess.Board = chess.Board()):
        results_dict = {}
        
        base_dict = self.get_base_bitboards_dict(board=board)
        
        results_dict.update(base_dict)

        return results_dict

    def get_numpy_bitboards(self,board):
        bitboards = self.get_all_bitboards(board=board)

        results = bitboards_to_array(np.array(list(bitboards.values())))

        return results
    




def bitboards_to_array(bb: np.ndarray) -> np.ndarray:
    bb = np.asarray(bb, dtype=np.uint64)[:, np.newaxis]
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(-1, 8, 8)


bc = Bitboard_Creator()

def get_all_bitboards_dict(board: chess.Board = chess.Board()):
    return bc.get_all_bitboards(board=board)

sample_bitboard_dict = get_all_bitboards_dict()