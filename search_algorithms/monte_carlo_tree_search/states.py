import numpy as np


class GameState:
    def __init__(self, board=None, current_player_id: int = 1):
        """
        Initializes a Tic-Tac-Toe game state.

        Args:
            board (np.ndarray, optional): A 3x3 numpy array representing the board.
                                          0 for empty, 1 for Player 1 (X), -1 for Player 2 (O).
                                          Defaults to an empty board.
            current_player_id (int): 1 for Player 1 (X), -1 for Player 2 (O).
        """
        if board is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            if not isinstance(board, np.ndarray) or board.shape != (3, 3) or not np.issubdtype(board.dtype, np.integer):
                raise ValueError("Board must be a 3x3 numpy array of integers.")
            self.board = board.copy()  # Ensure board is copied to maintain immutability

        if current_player_id not in [1, -1]:
            raise ValueError("current_player_id must be 1 (Player X) or -1 (Player O).")
        self.current_player_id = current_player_id  # 1 for X, -1 for O

    def get_possible_actions(self) -> list[tuple[int, int]]:
        """
        Returns a list of all empty (row, col) coordinates on the board,
        representing valid moves.
        """
        if self.is_terminal():
            return []

        actions = []
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    actions.append((r, c))
        return actions

    def apply_action(self, action: tuple[int, int]) -> 'GameState':
        """
        Applies a given action (placing a mark) and returns a new GameState.

        Args:
            action (tuple[int, int]): A (row, col) tuple indicating where to place the mark.

        Returns:
            GameState: A new GameState object after applying the action.
        """
        if not isinstance(action, tuple) or len(action) != 2 or \
                not all(isinstance(i, int) for i in action) or \
                not (0 <= action[0] < 3 and 0 <= action[1] < 3):
            raise ValueError("Action must be a (row, col) tuple within board bounds.")
        if self.board[action[0], action[1]] != 0:
            raise ValueError(f"Position {action} is already occupied.")

        new_board = self.board.copy()
        new_board[action[0], action[1]] = self.current_player_id

        # Switch to the other player for the next state
        next_player_id = -self.current_player_id

        return GameState(new_board, next_player_id)

    def is_terminal(self) -> bool:
        """
        Checks if the game has ended (win, loss, or draw).
        """
        return self.get_winner() != 0 or self.is_board_full()

    def get_winner(self) -> int:
        """
        Determines the winner of the game.
        Returns 1 if Player 1 wins, -1 if Player 2 wins, 0 if no winner yet or draw.
        """
        # Check rows
        for r in range(3):
            if np.all(self.board[r, :] == 1): return 1
            if np.all(self.board[r, :] == -1): return -1

        # Check columns
        for c in range(3):
            if np.all(self.board[:, c] == 1): return 1
            if np.all(self.board[:, c] == -1): return -1

        # Check diagonals
        if np.all(np.diag(self.board) == 1) or np.all(np.diag(np.fliplr(self.board)) == 1): return 1
        if np.all(np.diag(self.board) == -1) or np.all(np.diag(np.fliplr(self.board)) == -1): return -1

        return 0  # No winner

    def is_board_full(self) -> bool:
        """
        Checks if the board is completely filled.
        """
        return np.all(self.board != 0)

    def get_reward(self, player_id: int) -> float:
        """
        Returns the reward for the specified player from this terminal state.
        Args:
            player_id (int): The ID of the player for whom to calculate the reward (1 or -1).
        Returns:
            float: 1.0 for a win, 0.5 for a draw, 0.0 for a loss.
        """
        if not self.is_terminal():
            # Reward is only meaningful for terminal states in this model
            return 0.0

        winner = self.get_winner()
        if winner == player_id:
            return 1.0  # Current player wins
        elif winner == -player_id:
            return 0.0  # Current player loses (opponent wins)
        else:  # Draw
            return 0.5  # Draw

    def __eq__(self, other):
        """
        Compares two GameState objects for equality based on their board and current player.
        Crucial for MCTS node comparison and caching.
        """
        if not isinstance(other, GameState):
            return NotImplemented
        # Boards must be identical AND current player must be the same
        return np.array_equal(self.board, other.board) and self.current_player_id == other.current_player_id

    def __hash__(self):
        """
        Returns the hash of the GameState object.
        Crucial for MCTS node comparison and caching.
        """
        # A simple way to hash a numpy array is to convert it to a tuple of tuples.
        return hash((self.board.tobytes(), self.current_player_id))

    def __repr__(self):
        """
        Returns a string representation of the GameState object with a nicely formatted board.
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        board_str = "\n".join(
            " | ".join(symbols[self.board[r, c]] for c in range(3)) for r in range(3)
        )
        divider = "\n" + "- + - + -\n"
        formatted_board = divider.join(board_str.split("\n"))
        return (f"GameState(Player={'X' if self.current_player_id == 1 else 'O'},\n"
                f"Board:\n{formatted_board}\n)")