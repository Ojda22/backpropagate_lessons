
import numpy as np
from solvers import MCTSTicTacToe
from states import GameState

if __name__ == "__main__":
    # Starting from an empty board (Player X's turn) ---
    print("\n--- MCTS for Tic-Tac-Toe: Empty Board (Player X has turn ---")

    initial_board_empty = np.array([
        [0, 1, 0],
        [0, -1, 1],
        [-1, 0, 0]
    ], dtype=int) # An empty board has all zeros

    player_to_move = -1  # Player X starts, it is represented by 1 and Player O is represented by -1

    # a solver will have game state, root node that represents the initial state of the game
    # c_param is the exploration parameter for UCB1, which controls the balance between exploration and exploitation
    # current player_id is the player who is currently making a move
    # colver will be able to conduct MCTS search to find the best move for the current player
    # the functions of the solver are:
    #   - find_best_move(num_iterations) - runs MCTS for a given number of iterations and returns the best move
    #   - _select(node) - selects the best child node based on UCB1
    #   - _expand(node) - expands the node by creating a new child node for one of its untried actions
    #   - _simulate(node) - simulates a random game from the given node and returns the reward
    #   - _backpropagate(node, reward) - backpropagates the reward from the leaf node to the root node
    mcts_solver_empty = MCTSTicTacToe(initial_board=initial_board_empty,
                                      current_player_id=player_to_move,
                                      c_param=0) # c_param is the exploration parameter for UCB1

    # Run MCTS for a few thousand iterations, the more iterations, the better the move will be found
    best_move_empty = mcts_solver_empty.find_best_move(num_iterations=100000)

    print(f"\nRecommended move for Player {'X' if player_to_move == 1 else 'O'}: {best_move_empty}")
    if best_move_empty:
        # Show the board after the recommended move
        next_state = initial_board_empty.copy()
        next_state[best_move_empty[0], best_move_empty[1]] = player_to_move
        print("\nBoard after recommended move:")
        print(GameState(next_state, -player_to_move))  # Show the new state