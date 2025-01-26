import chess
import chess.pgn
import chess.engine
import random
from math import log,sqrt,e,inf
import ast
from chess_engine.src.model.config.config import Settings
import torch
import numpy as np


class node():
    def __init__(self):
        self.state = chess.Board()
        self.action = ''
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.v = 0


# white = 0, black = 1, stalemate = 2
class mcts():

    def __init__(self,board_analyzer,
                 ucb_constant:float = sqrt(2),
                 scores: list = [1.5,-1.5,0]) -> None:
        s = Settings()
        self.board_analyzer = board_analyzer
        # white, black, stalemate
        self.scores = scores
    
        self.c = ucb_constant
        random.seed(10)

    def ucb1(self,curr_node):
        ans = curr_node.v+self.c*(sqrt(log(curr_node.N+e+(10**-6))/(curr_node.n+(10**-10))))
        # print(ans)
        return ans

    def rollout(self,curr_node):


        if(curr_node.state.is_game_over()):
            board = curr_node.state
            if(board.result()=='1-0'):
                #print("h1")
                return (self.scores[0],curr_node)
            elif(board.result()=='0-1'):
                #print("h2")
                return (self.scores[1],curr_node)
            else:
                return (self.scores[2],curr_node)
        else:

            score_index = self.board_analyzer.use_model(curr_node.state)
            score = self.scores[score_index]

            return score, curr_node

    def expand(self,curr_node,white):
        if(len(curr_node.children)==0):
            return curr_node
        max_ucb = -inf
        if(white):

            max_ucb = -inf
            sel_child = None
            for i in curr_node.children:
                tmp = self.ucb1(i)
                if(tmp>max_ucb):

                    max_ucb = tmp
                    sel_child = i

            return(self.expand(sel_child,0))

        else:

            min_ucb = inf
            sel_child = None
            for i in curr_node.children:
                tmp = self.ucb1(i)
                if(tmp<min_ucb):

                    min_ucb = tmp
                    sel_child = i

            return self.expand(sel_child,1)

    def rollback(self,curr_node,reward):
        curr_node.n+=1
        curr_node.v+=reward
        while(curr_node.parent!=None):
            curr_node.N+=1
            curr_node = curr_node.parent
        return curr_node

    def mcts_pred(self,curr_node,over,
                  white,preferred_moves: list = None,
                  iterations: int = 10):
        if(over):
            return 0
        if preferred_moves:
            all_moves = []
            for i in preferred_moves:
                for move in ast.literal_eval(i):
                    all_moves.append(curr_node.state.uci(curr_node.state.parse_uci(move)))
        else:
            all_moves = [curr_node.state.uci(i) for i in list(curr_node.state.legal_moves)]
        map_state_move = dict()
        
        for i in all_moves:
            tmp_state = chess.Board(curr_node.state.fen())
            tmp_state.push_uci(i)
            child = node()
            child.state = tmp_state
            child.parent = curr_node
            curr_node.children.add(child)
            map_state_move[child] = i
            
        while(iterations>0):
            if(white):

                max_ucb = -inf
                sel_child = None
                for i in curr_node.children:
                    tmp = self.ucb1(i)
                    if(tmp>max_ucb):

                        max_ucb = tmp
                        sel_child = i
                ex_child = self.expand(sel_child,0)
                reward,state = self.rollout(curr_node=ex_child)
                curr_node = self.rollback(state,reward)
                iterations-=1
            else:

                min_ucb = inf
                sel_child = None
                for i in curr_node.children:
                    tmp = self.ucb1(i)
                    if(tmp<min_ucb):

                        min_ucb = tmp
                        sel_child = i

                ex_child = self.expand(sel_child,1)

                reward,state = self.rollout(ex_child)

                curr_node = self.rollback(state,reward)
                iterations-=1
        if(white):
            
            mx = -inf

            selected_move = ''
            for i in (curr_node.children):
                tmp = self.ucb1(i)
                if(tmp>mx):
                    mx = tmp
                    selected_move = map_state_move[i]
            return selected_move
        else:
            mn = inf

            selected_move = ''
            for i in (curr_node.children):
                tmp = self.ucb1(i)
                if(tmp<mn):
                    mn = tmp
                    selected_move = map_state_move[i]
            return selected_move



    def clear(self,node: node):
        if node.children is None:
            new_node = node.parent
            node = None
            self.clear(node=new_node)
        else:
            for child in node.children:
                self.clear(node=child)
            if node.parent is None:
                node = None
                del node
            return 0
        


    def set_ucb(self,ucb):
        self.c = ucb
    
    def set_scores(self,scores):
            self.scores = scores
    

    def mcts_best_move(self,board: chess.Board,iterations=100):
        random.seed(3141)
        root = node()
        root.state = board
        is_white_to_move = board.turn


        best_move_uci = self.mcts_pred(curr_node=root,over=board.is_game_over(),
                                white=is_white_to_move,
                                iterations=iterations)


        self.clear(node=root)

        return(best_move_uci)

