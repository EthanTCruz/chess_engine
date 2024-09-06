import chess

def count_xray_attacks(board):
    # Get the position of kings
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    
    # Initialize the x-ray attack counters
    xray_attacks_on_white_king = 0
    xray_attacks_on_black_king = 0

    # Calculate x-ray attacks for bishops, rooks, and queens
    for piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for attacker in board.pieces(piece_type, chess.WHITE):
            if board.is_attacked_by(chess.BLACK, attacker) and board.is_xray_attacker(attacker, black_king):
                xray_attacks_on_black_king += 1
        
        for attacker in board.pieces(piece_type, chess.BLACK):
            if board.is_attacked_by(chess.WHITE, attacker) and board.is_xray_attacker(attacker, white_king):
                xray_attacks_on_white_king += 1

    return xray_attacks_on_white_king, xray_attacks_on_black_king

# Example usage
board = chess.Board()
moves = ["e2e4","e7e5","f1b5","b8c6","g1f3","d7d6"]
for move in moves:
    board.push_uci(move)
xray_attacks_on_white_king, xray_attacks_on_black_king = count_xray_attacks(board)

print(f"X-ray attacks on white king: {xray_attacks_on_white_king}")
print(f"X-ray attacks on black king: {xray_attacks_on_black_king}")
