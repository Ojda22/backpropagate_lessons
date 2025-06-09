#
# Game Board Representation and Display
#

def create_board():
    """Initializes an empty Tic-Tac-Toe board."""
    return [' ' for _ in range(9)]  # 9 cells represented by spaces


def display_board(board) -> str:
    """Prints the Tic-Tac-Toe board in a human-readable format."""
    board_str = "\t-------------\n"
    board_str += f"\t| {board[0]} | {board[1]} | {board[2]} |\n"
    board_str += "\t-------------\n"
    board_str += f"\t| {board[3]} | {board[4]} | {board[5]} |\n"
    board_str += "\t-------------\n"
    board_str += f"\t| {board[6]} | {board[7]} | {board[8]} |\n"
    board_str += "\t-------------\n"

    # logging.info(board_str)
    return board_str

#
# Game Logic
#

def get_available_moves(board):
    """Returns a list of indices of empty cells on the board."""
    return [i for i, spot in enumerate(board) if spot == ' ']


def check_win(board, player):
    """
    Checks if the given player has won the game.
    The game is won if the player has three of their marks in a row, column, or diagonal.
    """
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
        [0, 4, 8], [2, 4, 6]  # Diagonals
    ]
    for combination in winning_combinations:
        if all(board[i] == player for i in combination): # check if all positions in the combination are occupied by the same player
            return True
    return False


def check_draw(board):
    """Checks if the game is a draw (no empty cells and no winner)."""
    return ' ' not in board and not check_win(board, 'X') and not check_win(board, 'O')

