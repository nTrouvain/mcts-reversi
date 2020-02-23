import os
import copy
import json
import time
import math

import numpy as np
from tqdm import tqdm
from Reversi import Board

# Méthode utilisant une constante d'exploration C_e 
def confidence_interval(confidence):
    def func(tree, node):
        interval = math.sqrt(confidence * (math.log(float(len(tree))) / float(node.N)))
        return node.score + interval
    return func

# Méthode de "moyenne harmonique" des victoires contre les pertes.
def harmonic_mean():
    def func(tree, node):
        loss = node.N - node.W
        return (node.W + 1) / (loss + 2)
    return func

class Node(object):
    
    def __init__(self, state, move, parent, terminal=False):
        self.move = move # comme définit par Reversi.py
        self.state = state # hash de l'état courant
        self.parent = parent # noeud parent
        self.childs = [] # noeuds successeurs
        self.N = 1e-8 # nombre de visites (constante epsilon pour éviter les problèmes de division par 0)
        self.W = 0 # nombre de victoire dans les noeuds successeurs terminaux
        self.score = 0.0 # estimateur du gain
        self.is_terminal = terminal
        
    def __repr__(self):
        return f"state: {self.state} N: {self.N} W: {self.W}, score:{self.score}"
    
    def update_score(self, reward):
        self.score = self.score + (reward - self.score) / self.N
        # mu = mu + (reward - mu) / nb_visites
        
class MCTS(object):
    
    BLACK = 1
    WHITE = 2
    
    def __init__(self, player=BLACK, score_method=None, confidence=None, nb_walk=50):
        
        self.player = player
        self.score_method = score_method or "score" # méthode de choix de l'action
        self.nb_walk = nb_walk # nombre d'explorations de l'arbre
        if confidence is not None:
            self.confidence = confidence # constante d'exploration
            
        self.score_func = self._set_score_func(score_method)
        
    def _set_score_func(self, method):
        # définit la méthode de choix de l'action
        if method == "confidence":
            return confidence_interval(self.confidence)
        elif method == "harmonic":
            return harmonic_mean()
        else:
            return lambda t, n: n.score
        
    def evaluate(self, original_board):
        # évaluation d'un état et sélection de la meilleure action
        self.tree = {}
        init_node = Node(hash(original_board._board.__str__()), None, None)
        self.tree[init_node.state] = init_node

        for _ in range(self.nb_walk):
            board = copy.deepcopy(original_board)
            self.walk(board, init_node)

        return init_node.childs, np.array([self.score_func(self.tree, node) for node in init_node.childs]).argmax()

    def walk(self, board, node):
        # exploration de l'arbre
        turn = 0
        
        # tant que le noeud est exploré ou n'est pas terminal
        while self.is_in_tree(node) and not(node.is_terminal):
            moves = board.legal_moves()
            next_nodes = self.find_next_nodes(board, node, moves)
            node.childs = next_nodes
            next_node = self.get_best_node(next_nodes) #sélection du noeud suivant
            board.push(next_node.move)
            node = next_node
            
            turn += 1
        
        # le noeud est nouveau
        self.tree[node.state] = node # ajout dans l'arbre
        result = self.random_walk(board, node) # "roll-out"
        self.propagate(node, result) # propagation du score vers les noeuds parents

    def random_walk(self, board, node):
        # roll-out
        while not(board.is_game_over()):
            moves = board.legal_moves()
            rand_move = moves[np.random.randint(len(moves))]
            board.push(rand_move)
        
        (whites, blacks) = board.get_nb_pieces()

        if whites > blacks:
            result = [0, 0, 1]
        elif whites < blacks:
            result = [0, 1, 0]
        else:
            result = [0, 0.5, 0.5]
        
        return result

    def propagate(self, node, result):
        curr_node = node
        # tant que l'on a pas remonté tout l'arbre
        while not(curr_node.parent is None):
            curr_node.N += 1
            curr_node.W += result[self.player] 
            curr_node.update_score(result[self.player]) # mise à jour du gain estimé
            curr_node = curr_node.parent

    def is_in_tree(self, node):
        tree_node = self.tree.get(node.state)
        if tree_node is None:
            return False
        return True

    def find_next_nodes(self, board, from_node, moves):
        nodes = []
        for m in moves:
            board.push(m)
            s = self.build_state(board, m)
            tree_node = self.tree.get(s)
            if tree_node is None:
                terminal = False
                if m[1] == -1 and m[2] == -1:
                    terminal = True
                nodes.append(Node(s, m, from_node, terminal))
            else:
                nodes.append(tree_node)
            board.pop()

        return nodes
    
    def get_best_node(self, nodes):
        # si les noeuds ont le même score on en prend un au hasard
        scores = [self.score_func(self.tree, n) for n in nodes]
        win_idx = np.argwhere(scores == np.amax(scores)).flatten()
        return nodes[np.random.choice(win_idx, 1)[0]]
        
           
    def build_state(self, board, from_move):
        # un etat est difini comme le hash du plateau de jeu + le mouvement à l'origine de cette configuration
        return hash(board._board.__str__() + from_move.__str__())


class RandomMover():
    # Joueur au hasard
    def __init__(self, player):
        self.player = player
        
    def evaluate(self, b):
        moves = b.legal_moves()
        return None, np.random.randint(len(moves))

def player_flipper(m_b, m_w):
    i = 0
    while True:
        if i % 2 == 0:
            i += 1
            yield m_b
        else:
            i += 1
            yield m_w
            

if __name__ == "__main__":
    # tests
    color = [0, "Black", "White"]    

    config = {
        "cls-rand": [
                MCTS(player=MCTS.BLACK),
                RandomMover(player=MCTS.WHITE)
        ],
        "har-rand": [
                MCTS(player=MCTS.BLACK, score_method="harmonic"),
                RandomMover(player=MCTS.WHITE)
        ],
        "cf005-rand": [
                MCTS(player=MCTS.BLACK, score_method="confidence", confidence=0.05),
                RandomMover(player=MCTS.WHITE)
        ],
        "cf01-rand":[
                MCTS(player=MCTS.BLACK, score_method="confidence", confidence=0.1),
                RandomMover(player=MCTS.WHITE)
        ],
        "cls-har": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="harmonic")
        ],
        "cls-cf001": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.01)
        ],
        "cls-cf005": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.05)
        ],
        "cls-cf01": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.1)
        ],
        "cls-cf03": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.3)
        ],
        "cls-cf05": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.5)
        ],
        "cls-cf07": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.7)
        ],
        "cls-cf09": [
            MCTS(player=MCTS.BLACK),
            MCTS(player=MCTS.WHITE, score_method="confidence", confidence=0.9)
        ]
    }
    
    
    for name, cfs in config.items():
        
        reports = []
        for _ in tqdm(range(30), f"Match {name}"):
            turn = {
                "moves": []
            }
            
            b = Board(8)
            players = player_flipper(*cfs)
            
            while not(b.is_game_over()):
                player = next(players)
                move_idx = player.evaluate(b)
                move = b.legal_moves()[move_idx[1]]
                b.push(move)
                turn["moves"].append(move)
            turn["outcome"] = b.get_nb_pieces()
            reports.append(turn)

        with open(os.path.join("tests", f"{name}.json"), "w+") as f:
            json.dump(reports, f)
        

