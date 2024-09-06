import chess
from math import log, sqrt, inf
import random
import time
import sys
sys.setrecursionlimit(10000)


C = sqrt(2)
class Node():
    def __init__(self):
        self.state = chess.Board()
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0

def ucb(node: Node):
    value = node.v/node.n + C * sqrt(log(node.N + 1)/(node.n + (10**-10)))
    return value
def selection(node: Node):
    highest_ucb = -inf
    highest = node
    for child in node.children:
        if child.children:
            child, child_ucb = selection(node = child)
        else:
            child_ucb = ucb(child)
        if child_ucb > highest_ucb and child.state.result() == '*':
            highest_ucb = child_ucb
            highest = child
    return highest, highest_ucb
            
def expand(node: Node):
    board = node.state.copy()
    
    #Should be replaced with highest scored move with nn
    move = random.choice(list(board.legal_moves))
    board.push(move)
    child = Node()
    child.state = board
    child.N = node.N
    child.parent = node
    node.children.add(child)
    return child

def simulation(node: Node,white: bool):
    board = node.state.copy()
    while board.result() == '*':
        move = random.choice(list(board.legal_moves))
        #Replace ove with highest for white or black
        board.push(move)
    if board.result() == "1-0":
        if white:
            return 1
        else:
            return 0
    elif board.result() == "0-1":
        if white:
            return 0
        else:
            return 1
    elif board.result() == "1/2-1/2":
        return 0.5
    else:
        raise Exception("Escaped loop without winner",board.result())
        
def backpropagate(node: Node,reward: int):

    node.v += reward
    node.n += 1
    
    current = node.parent
    while current is not None:
        current.N += 1  
        current.v += reward  
        current = current.parent
    return node

def shallow_find_best_move(node: Node):
    if node.children is None:
        move = random.choice(list(node.state.legal_moves))
        return move
    highest_ucb = -inf
    highest = node
    for child in node.children:
        child_ucb = ucb(child)
        if child_ucb > highest_ucb:
            highest_ucb = child_ucb
            highest = child
    return highest.state.move_stack[-1]

def mcst_timed(board: chess.Board,time_limit: float):
    root = Node()
    root.state = board
    white = board.turn
    start_time = time.time()
    end_time = start_time + time_limit
    while time.time() <= end_time:
        selected_node = selection(node=root)[0]
        expanded_node = expand(node=selected_node)
        reward = simulation(node=expanded_node,white=white)
        backpropagate(node=expanded_node,reward=reward)
    move = shallow_find_best_move(node=root)
    return move

# random.seed(3141)
board = chess.Board()
board.push_san("e4")
board.push_san("e5")
move = mcst_timed(board=board,time_limit=30)
print(move)