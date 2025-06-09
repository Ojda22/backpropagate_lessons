import logging
import math
import uuid

import graphviz

from search_algorithms.minmax.utils import display_board

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='> %(message)s')

class MoveNode:
    """
    Represents a move graph for the Tic-Tac-Toe game.
    This class can be extended to include more advanced features like move history, etc.
    Currently, it is a placeholder for potential future enhancements.
    """

    def __init__(self, board, turn='AI'):
        self._move_score = -math.inf # We want to maximize the score, so we start with negative infinity
        self.board = board.copy() # The current state of the Tic-Tac-Toe board, represented as a list of 9 spaces, and each space can be 'X', 'O', or ' ' (empty)
        self._move = '?'  # '?' indicates an uninitialized move, can be replaced with an integer later
        self._depth = -1  # Depth of the move in the decision tree, can be used for advanced features
        self._turn = turn
        self._potential_moves: list[MoveNode] = []  # List to hold potential moves, each represented as a Move object
        self._node_id = str(uuid.uuid4())

    @property
    def node_id(self):
        """Returns the unique identifier for the move node."""
        return self._node_id

    @node_id.setter
    def node_id(self, node_id):
        """Setter for node_id with optional validation"""
        self._node_id = node_id

    @property
    def turn(self):
        """Returns the turn of the player making the move."""
        return self._turn

    @turn.setter
    def turn(self, turn):
        """Setter for turn with optional validation"""
        if turn not in ['AI', 'Human']:
            raise ValueError("Turn must be either 'AI' or 'Human'.")
        self._turn = turn

    @property
    def depth(self):
        """Returns the depth of the move in the decision tree."""
        return self._depth

    @depth.setter
    def depth(self, depth):
        """Setter for depth with optional validation"""
        if not isinstance(depth, int) or depth < 0:
            raise ValueError("Depth must be a non-negative integer.")
        self._depth = depth

    @property
    def move(self):
        """Returns the current move."""
        return self._move

    @move.setter
    def move(self, move):
        """Setter for move with optional validation"""
        if not (isinstance(move, int) and 0 <= move <= 8) and move != '?':
            raise ValueError("Move must be an integer between 0 and 8 or ? character.")
        self._move = move

    @property
    def move_score(self):
        """Returns the score of the move."""
        return self._move_score

    @move_score.setter
    def move_score(self, score):
        """Setter for move_score with optional validation"""
        if not isinstance(score, (int, float)):
            raise ValueError("Score must be a number.")
        self._move_score = score

    def __str__(self):
        """
        Node representation for the decision graph.
        Has only node id
        :return:
        """
        return f"Node {self._node_id}"

    def label(self):
        """
        Returns a label for the node, used in the decision graph.
        This method can be extended to include more information if needed.
        """
        return (f"Turn: {self.turn}\nMove: {self.move} \nScore: {self.move_score}\n"
                f"{display_board(board=self.board)}\n"
                f"Depth: {self.depth}")  # Label for the node in the decision graph

    def add_potential_move(self, available_move: 'MoveNode'):
        """
        A method to add an available move as a potential move to the current move graph.
        """
        self._potential_moves.append(available_move)  # Add the available move to the potential moves list

    def get_potential_moves(self):
        """Returns the list of available moves."""
        return self._potential_moves

    def export_decision_graph(self):
        """
        Exports the decision graph to a png file.
        This method takes the Move node and its potential moves and creates a visual representation of the decision graph.
        """
        graph = graphviz.Digraph(format='svg', graph_attr={'rankdir': 'TD'})  # TD = top down, LR = left to right
        graph.node(name=str(self), label=self.label())  # Add the current move node
        self._add_edges(graph, self)  # Recursively add edges for potential moves
        graph.view()


    def _add_edges(self, graph, parent_node):
        """
        Recursively adds edges to the decision graph for each potential move.
        """
        for move in parent_node.get_potential_moves():
            graph.node(name=str(move), label=move.label())  # Add the move node to the graph
            graph.edge(str(parent_node), str(move))  # Add an edge from the parent node to the move node
            if move.get_potential_moves():
                move._add_edges(graph, move)

    def export_decision_graph_to_console(self):
        """
        Exports the decision graph to the console using a text-based representation.
        This method iterates through the Move node and its potential moves,
        printing a hierarchical view of the decision graph.
        """
        self._print_graph_recursive(self, 0)

    def _print_graph_recursive(self, node, indent_level):
        """
        Recursively prints the decision graph to the console with indentation.
        """
        indent = "---" * indent_level  # Indentation for the current level
        logging.info(f"{indent}{'root' if node.depth == -1 else node.depth} (depth) | Move: {node.move}, Score: {node.move_score} | Turn: {node.turn} | NodeId: {node.node_id}")  # Print the current node's details

        for potential_move in node.get_potential_moves():
            self._print_graph_recursive(potential_move, indent_level + 1)