import logging
import math

from search_algorithms.minmax.move_node import MoveNode
from search_algorithms.minmax.utils import check_win, check_draw, get_available_moves, create_board, display_board

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='> %(message)s')

#
# Minimax Algorithm (Adversarial Search)


def minimax(board, depth, is_maximizing_player, move_node=None):
    """
    The Minimax algorithm implementation.

    Args:
        board (list): The current state of the Tic-Tac-Toe board.
        depth (int): The current depth of the search tree.
        is_maximizing_player (bool): True if the current player is the AI (maximizing),
                                     False if it's the human (minimizing).
        move_node (MoveNode): The current move node in the move graph, used for tracking potential moves.

    Returns:
        int: The score of the best possible move.
    """
    if check_win(board, 'O'):  # AI wins
        return 10 - depth  # Prioritize winning faster
    elif check_win(board, 'X'):  # Human wins
        return depth - 10  # Prioritize losing slower
    elif check_draw(board):  # Draw
        return 0

    if is_maximizing_player:
        best_score = -math.inf  # Initialize with negative infinity
        for move in get_available_moves(board):
            board[move] = 'O'  # Make the move for AI, which is maximizing player, and calculate the score
            available_move = MoveNode(board)
            available_move.move = move  # Set the current move in the move graph
            available_move.depth = depth + 1  # Increment the depth for the move graph
            available_move.turn = "AI"  # Set the turn for the move graph to AI's turn
            move_node.add_potential_move(available_move)
            score = minimax(board, depth + 1, False, move_node=available_move)  # Recurse for opponent
            available_move.move_score = score  # Set the score in the move graph
            board[move] = ' '  # Undo the move (backtrack)
            best_score = max(best_score, score)
        return best_score
    else:  # Minimizing player (human)
        best_score = math.inf  # Initialize with positive infinity
        for move in get_available_moves(board):
            board[move] = 'X'  # Make the move for human
            available_move = MoveNode(board)
            available_move.move = move  # Set the current move in the move graph
            available_move.depth = depth + 1  # Increment the depth for the move graph
            available_move.turn = "Human"  # Set the turn for the move graph to Human's turn
            move_node.add_potential_move(available_move)
            score = minimax(board, depth + 1, True, move_node=available_move)  # Recurse for AI
            available_move.move_score = score
            board[move] = ' '  # Undo the move (backtrack)
            best_score = min(best_score, score)
        return best_score


#
# AI Player Logic
#

def get_best_ai_move(board):
    """
    Uses the Minimax algorithm to find the best move for the AI.
    """
    best_score = -math.inf # Initialize with negative infinity, we want to maximize the score
    best_move = -1 # Initialize with an invalid move

    decision_node = MoveNode(board)  # Create a move graph for the current board state

    for idx, move in enumerate(get_available_moves(board)):  # Iterate through all available moves (empty cells)
        board[move] = 'O'  # Try the move
        available_move = MoveNode(board)  # Create a move graph for the available move
        available_move.turn = "AI"  # Default is 'AI', but we set it explicitly for clarity
        available_move.move = move # Set the current move in the move graph
        available_move.depth = 0  # Set the depth for the move graph to 0, as this is the root of the decision tree
        decision_node.add_potential_move(available_move)  # Add the move graph to the move node
        score = minimax(board, 0, False, available_move)  # Calculate score for this move, and recursively call minimax to evaluate the move and follow the game tree
        board[move] = ' '  # Undo the move
        available_move.move_score = score  # Set the score in the move graph

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, decision_node  # Return the best move and the move graph for potential future use

#
# Game Loop
#

def play_game(human_has_first_move):
    """Main function to run the Tic-Tac-Toe game."""
    board = create_board() # Initialize the game board, which is a list of 9 spaces
    current_player = 'O' if human_has_first_move == 'no' else 'X' # Human starts as 'X' and AI is 'O'

    logging.info("Welcome to Tic-Tac-Toe!")
    logging.info("You are 'X', the AI is 'O'.")
    logging.info(display_board(board)) # Display the initial empty board (created template how the board looks like)

    while True: # we loop until there is a winner or a draw
        if current_player == 'X':  # Human's turn
            try:
                move = input("Enter your move (0-8) - or (resign): ") # 0-8 corresponds to the board cells

                if move == 'resign':  # Check if the player wants to resign
                    logging.info("You resigned. AI wins!")
                    break

                move = int(move)  # Convert input to integer

                if move not in get_available_moves(board): # Check if the move is valid, the move is valid if it is on empty and existing cell
                    logging.info("Invalid move. Please choose an empty cell (0-8).")
                    continue
                board[move] = 'X'

                logging.info("\t---Current Board State---")
                logging.info(display_board(board))  # Display the board after each move
            except ValueError:
                logging.info("Invalid input. Please enter a number between 0 and 8.")
                continue
        else:  # AI's turn
            logging.info("AI is thinking...")
            move, move_node = get_best_ai_move(board) # Get the best move for the AI using Minimax Search Algorithm
            board[move] = 'O'
            logging.info(f"AI chose move: {move}")

            logging.info("\t---Current Board State---")
            logging.info(display_board(board))  # Display the board after each move

            # Display the trace of the decision graph
            export_traces = input("Do you want to see the AI decision graph? (yes/no): ").strip().lower()
            if export_traces == 'yes':
                move_node.export_decision_graph()  # Export the decision graph to a file
            elif export_traces not in ['yes', 'no']:
                logging.info("Invalid input, decision graph will not be exported.")

        if check_win(board, current_player):
            logging.info(f"Player {current_player} wins!")
            break
        elif check_draw(board):
            logging.info("It's a draw!")
            break

        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'


if __name__ == "__main__":
    logging.info("Starting Tic-Tac-Toe game...")
    human_has_first_move = input("Do you want to start first? (yes/no): ").strip().lower()
    if human_has_first_move not in ['yes', 'no']:
        logging.info("Invalid input, AI will start, hold on :)")
        human_has_first_move = 'no'
    play_game(human_has_first_move)