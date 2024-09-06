import chess
from math import log, sqrt, inf
import random
import time
import sys
from chess_engine.src.model.classes.pretorch_files.potential_board_populator import populator
from chess_engine.src.model.config.config import Settings
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

class mcts():
    def __init__(self,neuralNet,constant:float = None) -> None:
        s = Settings()
        self.nn = neuralNet
        if constant is None:
            self.c = s.UCB_Constant
        baseBoard = chess.Board()
        baseDepth = 1
        self.populator = populator(board=baseBoard,score_depth=baseDepth)


    def ucb(self,node: Node):
        value = node.v/node.n + C * sqrt(log(node.N + 1)/(node.n + (10**-10)))
        return value

    def selection(self,node: Node):
        highest_ucb = -inf
        highest = node
        for child in node.children:
            if child.children:
                child, child_ucb = self.selection(node = child)
            else:
                child_ucb = self.ucb(child)
            if child_ucb > highest_ucb and child.state.result() == '*':
                highest_ucb = child_ucb
                highest = child
        return highest, highest_ucb
                
    def expand(self,node: Node):
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

    def simulation_sm_return(self,node: Node,white: bool,max_depth: int=100):
            board = node.state.copy()
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
            elif board.result() == '*':
                score = self.nn.score_board(board)
                white_score = float(score['white'])
                black_score = float(score['black'])
                stalemate_score = float(score['stalemate'])
                    # Generate a random number between 0 and 1
                rand_num = random.random()

                # Determine which score range rand_num falls into
                if rand_num <= white_score:
                    if white:
                        return 1
                    else:
                        return 0
                elif rand_num <= white_score + black_score:
                    if white:
                        return 0
                    else:
                        return 1

                else:
                    return 0.5

            else:
                raise Exception("Escaped loop without winner",board.result())

    def simulation(self,node: Node,white: bool,max_depth: int=100):
        board = node.state.copy()
        i = 0
        while board.result() == '*' and i < max_depth:
            i += 1
            # self.populator.set_depth_and_board(depth=1,board=board)
            # self.populator.get_all_moves()
            # moves = self.populator.return_total_moves()

            # scores = self.nn.score_moves(total_moves=moves)
            # if white:
            #     scores = scores.sort_values(by=['white'],ascending=False)
            # else:
            #     scores = scores.sort_values(by=['black'],ascending=False)
            # move_series = scores.head(1)['moves(id)'] # This gives you the Series
            # move_list = move_series.iloc[0] # Access the first item in the series, which is a list in string format
            # move = eval(move_list)[0] # Convert string list to actual list and get the first element
            # board.push_uci(move)

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
        elif board.result() == '*':
            score = self.nn.score_board(board)
            if white:
                score = float(score['white'])
            else:
                score = float(score['black'])
            return score
        else:
            raise Exception("Escaped loop without winner",board.result())
            
    def backpropagate(self,node: Node,reward: int):

        node.v += reward
        node.n += 1
        
        current = node.parent
        while current is not None:
            current.N += 1  
            current.v += reward  
            current = current.parent
        return node

    def shallow_find_best_move(self,node: Node):
        if node.children is None:
            move = random.choice(list(node.state.legal_moves))
            return move
        highest_ucb = -inf
        highest = node
        for child in node.children:
            child_ucb = self.ucb(child)
            if child_ucb > highest_ucb:
                highest_ucb = child_ucb
                highest = child
        return highest.state.move_stack[-1]

    def mcst_timed(self,board: chess.Board,time_limit: float):
        self.nn.load_scaler()
        root = Node()
        root.state = board
        white = board.turn
        start_time = time.time()
        end_time = start_time + time_limit
        while time.time() <= end_time:
            selected_node = self.selection(node=root)[0]
            expanded_node = self.expand(node=selected_node)
            # reward = self.simulation(node=expanded_node,white=white)
            reward = self.simulation_sm_return(node=expanded_node,white=white)
            self.backpropagate(node=expanded_node,reward=reward)
        move = self.shallow_find_best_move(node=root)
        return move

