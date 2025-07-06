import time
import copy
import random
import numpy as np  # For initial board, though GameState handles it now

from nodes import MCTSNode
from states import GameState


class MCTSTicTacToe:
    def __init__(self, initial_board: np.ndarray = None, current_player_id: int = 1, c_param: float = 1.4):
        """
        Initializes the MCTS Solver for Tic-Tac-Toe.

        Args:
            initial_board (np.ndarray, optional): The starting Tic-Tac-Toe board.
                                                  Defaults to an empty board.
            current_player_id (int): The player whose turn it is at the start (1 for X, -1 for O).
            c_param (float): The exploration parameter for UCB1.
        """
        # game state has the board and current player. All the logic for the game is in GameState.
        # all functions that are related to the game state are in GameState.
        # actions are:
        #   - get_possible_actions() - returns a list of possible actions (row, col) for the current player
        #   - apply_action(action) - applies the action to the current state and returns a new GameState
        #   - is_terminal() - checks if the game is over (win, loss, draw)
        #   - get_reward(player_id) - returns the reward for the given player_id (1.0 for win, 0.5 for draw, 0.0 for loss)
        #   - get_winner() - returns the winner of the game (1 for X, -1 for O, 0 for draw)
        initial_state = GameState(board=initial_board, current_player_id=current_player_id)
        self.root = MCTSNode(initial_state) # Create the root node with the initial game state
        # the idea is to have a tree of nodes, where each node represents a game state
        # and the edges represent the actions taken to reach that state.
        # we aim to find the best move for the current player by simulating many games
        self.c_param = c_param
        # Store the ID of the player for whom the MCTS is running (the AI player)
        self.ai_player_id = current_player_id

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Phase 1: Selection.
        Traverses the tree from the given node (usually the root) by repeatedly
        selecting the best child based on the UCB1 formula, until a node
        that is not fully expanded or is terminal is reached.

        Args:
            node (MCTSNode): The starting node for selection.

        Returns:
            MCTSNode: The selected node (either terminal or not fully expanded).
        """
        current_node = node  # Start from the given node (initially, the MCTS root)

        # Loop continues as long as the current node is NOT terminal AND is fully expanded.
        # This means we continue selecting as long as there's a path *already built*
        # that we can traverse deeper into.
        while not current_node.is_terminal_node() and current_node.is_fully_expanded():
            # If the node is fully expanded, it means all its possible moves already
            # have corresponding child nodes in the tree. We must select one of them.
            current_node = current_node.best_child(self.c_param)
            # The best_child() method uses the UCB1 formula to pick.
        return current_node  # Returns the node where selection stops

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Phase 2: Expansion.
        If the selected node is not terminal and not fully expanded,
        create a new child node for one of its untried actions.
        """
        if node.is_terminal_node():
            return node  # Cannot expand a terminal node

        try:
            new_child_node = node.expand()
            return new_child_node
        except RuntimeError:
            # If no untried actions (e.g., board is full or no moves, but not yet terminal by logic),
            # return the node itself. This signifies it's "effectively" expanded for this iteration.
            return node

    def _simulate(self, node: MCTSNode) -> float:
        """
        Phase 3: Simulation (Rollout).
        Performs a random playout from the given node's state until a terminal state is reached.
        Returns the reward *from the perspective of the AI player (self.ai_player_id)*.
        """
        current_sim_state = copy.deepcopy(node.state)  # Start simulation from a copy

        while not current_sim_state.is_terminal():
            possible_actions = current_sim_state.get_possible_actions()
            if not possible_actions:
                # This should ideally not happen if is_terminal() covers all end conditions.
                # If board is full but no winner (draw), it's terminal.
                # If there are empty spots but no moves possible, implies bug.
                break  # Should only break if logic dictates a pseudo-terminal state

            random_action = random.choice(possible_actions)
            current_sim_state = current_sim_state.apply_action(random_action)

        # Get the reward from the perspective of the player *who started the simulation*.
        # When we backpropagate, we'll invert this for the *other* player's nodes.
        # But `GameState.get_reward` is already designed to give reward for a *specific* player.
        # So, the reward here should be for the player whose turn it was *at the `node.state`*.
        # This means, `node.state.current_player_id`.
        return current_sim_state.get_reward(node.state.current_player_id)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Phase 4: Backpropagation.
        Updates the visit counts and total rewards of all nodes from the given
        node up to the root of the tree.
        Crucially, for alternating-turn games, the reward needs to be inverted
        for the opposing player's nodes as it propagates up.
        """
        current_node = node
        while current_node is not None:
            current_node.update(reward)
            # For Tic-Tac-Toe, reward alternates.
            # If current_node.state.current_player_id wins, then current_node.parent.state.current_player_id loses.
            # So, flip the reward for the parent's player perspective.
            # Reward of 1.0 (win) becomes 0.0 (loss) for opponent.
            # Reward of 0.0 (loss) becomes 1.0 (win) for opponent.
            # Reward of 0.5 (draw) remains 0.5.
            if current_node.parent is not None:
                # Reward for current player's win (1.0) is opponent's loss (0.0)
                # Reward for current player's loss (0.0) is opponent's win (1.0)
                # Reward for draw (0.5) stays 0.5
                reward = 1.0 - reward  # This correctly flips 1.0 to 0.0, 0.0 to 1.0, and keeps 0.5 at 0.5
            current_node = current_node.parent

    def find_best_move(self, num_iterations: int) -> tuple[int, int]:
        """
        Runs the MCTS algorithm to find the best move for the current player.

        Args:
            num_iterations (int): The number of MCTS iterations to perform.

        Returns:
            tuple[int, int]: The (row, col) action recommended by MCTS.
        """
        if not isinstance(num_iterations, int) or num_iterations <= 0:
            raise ValueError("num_iterations must be a positive integer.")
        if self.root.is_terminal_node():
            raise RuntimeError("Cannot find a move for a terminal game state.")

        print(f"MCTS for Player {'X' if self.ai_player_id == 1 else 'O'} starting with {num_iterations} iterations...")
        start_time = time.time()

        # Run MCTS for the specified number of iterations
        # Each iteration consists of:
        #   1. Selection: Traverse the tree to find a node to simulate from
        #   2. Expansion: If the node is not terminal, expand it by creating a new child node
        #   3. Simulation: Simulate a random game from the expanded node
        #   4. Backpropagation: Update the nodes in the path from the simulated node to the root
        for i in range(num_iterations):
            selected_node = self._select(self.root) # 1. Selection

            # 2. Expansion
            # If the selected node is terminal, we don't expand.
            # We just simulate from it (which will return its terminal reward).
            if not selected_node.is_terminal_node():
                node_to_simulate_from = self._expand(selected_node)
            else:
                node_to_simulate_from = selected_node

            # 3. Simulation
            # The reward is for the player whose turn it was *at the node_to_simulate_from*.
            reward = self._simulate(node_to_simulate_from)

            # 4. Backpropagation
            # The reward propagates upwards, flipping for alternating player nodes.
            self._backpropagate(node_to_simulate_from, reward)

        end_time = time.time()
        print(f"MCTS finished in {end_time - start_time:.2f} seconds.")

        # After all iterations, select the best child of the root node
        # based on the most visits. This is the recommended move.
        if not self.root.children:
            # This should ideally not happen if there are possible moves from the root.
            # If the root is not terminal but has no children, it means get_possible_actions
            # from root state returned nothing, which would be a bug or impossible game state.
            print("Warning: Root node has no children after MCTS. No moves found.")
            return None

        best_child = max(self.root.children.values(), key=lambda child: child.visits)

        print("\n--- Top Recommended Moves (by Visits) ---")
        sorted_children = sorted(self.root.children.values(), key=lambda c: c.visits, reverse=True)
        for i, child in enumerate(sorted_children[:min(len(sorted_children), 5)]):
            print(
                f"  {i + 1}. Move: {child.action}, Visits: {child.visits}, Total Award: {child.total_reward}, Avg Reward: {child.total_reward / child.visits:.2f}")

        return best_child.action