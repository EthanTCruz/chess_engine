import chess
from chess_engine.src.model.config.config import Settings
import chess
import chess.syzygy
from math import inf




class endgamePicker():

    def __init__(self,**kwargs) -> None:
        self.set_parameters(kwargs=kwargs)

    def set_parameters(self,kwargs):
        s = Settings()
        if "endgame_table" not in kwargs:
            self.endgame_table = s.endgame_table
        else:
            self.endgame_table = kwargs["endgame_table"]

        if "minimumEndgamePieces" not in kwargs:
            self.endgameAmount = s.minimumEndgamePieces
        else:
            self.endgameAmount = kwargs["minimumEndgamePieces"]



    def open_tables(self):
        self.tablebase = chess.syzygy.Tablebase()
        self.tablebase.add_directory(self.endgame_table)
    
    def close_tables(self):
        self.tablebase.close()

    def has_move_dtz(self,board: chess.Board):
        results = 0
        result = self.tablebase.probe_dtz(board)
        if result is not None:
            results = result
        return results


    def find_endgame_best_move_dtz(self,board: chess.Board):
        temp = board.copy()

        if self.count_pieces(board=temp) <= self.endgameAmount:
            result = self.tablebase.probe_dtz(board)
            if result is not None:
                if result != 0:
                    best_probe = -inf
                    best_move = None
                    for move in temp.legal_moves:
                        temp.push(move)
                        probe = self.tablebase.probe_dtz(temp)
                        if (temp.halfmove_clock - 100) <= probe < 0:
                            if probe > best_probe:
                                if probe == -1:
                                    return move
                                else:
                                    best_probe = probe
                                    best_move = move

                        temp.pop()

                        if best_move:

                            return best_move
                else:

                    return 0
            else:
                return 0
        else:
            return 0

    def has_move(self,board: chess.Board):
        self.open_tables()
        result = self.tablebase.probe_wdl(board)
        self.close_tables()
        if result is not None:
            return result
        else:
            return 0


    def find_endgame_best_move(self,board: chess.Board):
        temp = board.copy()

        if self.count_pieces(board=temp) <= self.endgameAmount:
            result = self.has_move(board=temp)
            if result != 0:
                best_move = []
                best_dtz_probe = -inf

                for move in temp.legal_moves:
                    temp.push(move)

                    dtz_probe = self.tablebase.probe_dtz(temp)

                    if ((temp.halfmove_clock - 100) <= dtz_probe < 0):
                        if dtz_probe > best_dtz_probe:
                            if dtz_probe == -1:
                                return [str(move)]
                            else:
                                best_dtz_probe = dtz_probe
                                best_move = [str(move)]
                        elif dtz_probe == best_dtz_probe:
                            best_move.append(str(move))

                    temp.pop()

                if len(best_move) != 0:

                    return best_move
                else:

                    return []
            else:
                return []
        else:
            return []

    def non_drawing_moves(self,board: chess.Board):
        temp = board.copy()
        non_drawing_moves = []
        best_moves = []
        if self.count_pieces(board=temp) <= self.endgameAmount:
            result = self.has_move(board=temp)
            if result != 0:

                for move in temp.legal_moves:
                    temp.push(move)

                    dtz_probe = self.tablebase.probe_dtz(temp)

                    if  ((temp.halfmove_clock - 100) <= dtz_probe <= -1) or ((100 - temp.halfmove_clock) >= dtz_probe >= 1):

                        if dtz_probe == -1:
                            return [str(move)]
                        else:
                            non_drawing_moves.append(move)
                    temp.pop()

        return non_drawing_moves


    def count_pieces(self,board: chess.Board):
        pieces = board.piece_map()
        return len(pieces)
    
    def endgame_status(self,board: chess.Board):
        temp = board.copy()
        result = 0
        result = self.has_move_dtz(board=temp)
        return result