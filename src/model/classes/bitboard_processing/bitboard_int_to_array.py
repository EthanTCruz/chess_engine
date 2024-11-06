import numpy as np
def bitboard_to_matrix(bitboard):
    return np.array([(bitboard >> shift) & 1 for shift in range(64)]).reshape(8, 8)

def create_cnn_input(bitboards):
    layers = []
    for bb in bitboards:  # Ensure consistent order
        matrix = bitboard_to_matrix(int(bb))
        # print(matrix)
        layers.append(matrix)
    return np.stack(layers)