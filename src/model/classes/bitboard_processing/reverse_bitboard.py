from chess_engine.src.model.classes.bitboard_processing.bitboard_creator import bitboards_to_array, sample_bitboard_dict, Bitboard_Creator
import chess




bb_keys = list(sample_bitboard_dict.keys())

def numpy_bitboards_to_board(bitboards):

    # bitboards = np.zeros((12, 8, 8), dtype=int)
    bitboards = bitboards.copy()
    
    fen_symbols = []
    
    for key in bb_keys:
        color, piece = key.split(' ')
        if piece == "knight":
            piece = "n"
        if color == 'white':
             piece = piece.upper()
        fen_symbols.append(piece[0])
    
    # # FEN symbols for each bitboard index
    # fen_symbols = ['P', 'N', 'B', 'R', 'Q', 'K',  # White pieces
    #                'p', 'n', 'b', 'r', 'q', 'k']  # Black pieces
    
    # Initialize an empty board
    board = [['' for _ in range(8)] for _ in range(8)]
    
    # Populate the board using bitboards
    for idx, bitboard in enumerate(bitboards):
        for row in range(8):
            for col in range(8):
                if bitboard[row, col] == 1:
                    board[row][col] = fen_symbols[idx]
    
    # Convert board to FEN string
    fen_rows = []
    for row in board:
        fen_row = ""
        empty_count = 0
        for cell in row:
            if cell == '':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += cell
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    # Join rows with '/'
    fen_position = "/".join(fen_rows)
    
    # Add additional FEN fields (default turn, castling, etc.)
    full_fen = fen_position + " w - - 0 1"
    
    board_obj = chess.Board(full_fen)
    
    return board_obj
