import math
import random

from states import GameState

class MCTSNode:
    def __init__(self, state: GameState, parent=None, action=None):
        """
        Initializes a new node in the MCTS tree.

        Args:
            state (GameState): The Tic-Tac-Toe game state this node represents.
            parent (MCTSNode, optional): The parent node in the tree. None for the root node.
            action (tuple[int, int], optional): The action (row, col) that led to this state from the parent.
        """
        self.state = state
        self.parent = parent
        self.action = action  # The action (move) that led to this state from its parent

        self.children = {}  # Dictionary to store child nodes: {action: MCTSNode}
        self.visits = 0  # Number of times this node has been visited
        # total_reward will accumulate the sum of rewards (1.0 win, 0.5 draw, 0.0 loss)
        # from the perspective of the player who's turn it is *at this node's state*.
        self.total_reward = 0.0

        # Untried actions: these are actions that haven't yet been explored
        # (i.e., new child nodes haven't been created for them)
        self.untried_actions = None  # Will be initialized lazily

    def is_terminal_node(self) -> bool:
        """
        Checks if the node's state is a terminal state.
        """
        return self.state.is_terminal()

    def expand(self) -> 'MCTSNode':
        """
        Expands the node by creating a new child node for one of its untried actions.

        Returns:
            MCTSNode: The newly created child node.

        Raises:
            RuntimeError: If there are no untried actions to expand.
        """
        if self.untried_actions is None:
            self.untried_actions = self.state.get_possible_actions()
            # Shuffle to ensure random selection if UCB1 values are tied initially
            # Also provides a good spread for initial explorations
            random.shuffle(self.untried_actions)

        if not self.untried_actions:
            raise RuntimeError(
                "Cannot expand a node with no untried actions. This node might be terminal or fully expanded.")

        # Pick one untried action
        action = self.untried_actions.pop()

        # Apply the action to get the new state
        new_state = self.state.apply_action(action)

        # Create a new child node
        child_node = MCTSNode(new_state, parent=self, action=action)
        self.children[action] = child_node

        return child_node

    def is_fully_expanded(self) -> bool:
        """
        Checks if all possible actions from this node's state have been explored
        (i.e., child nodes have been created for them).
        """
        if self.untried_actions is None:  # Initialize if not already done
            self.untried_actions = self.state.get_possible_actions()
            random.shuffle(self.untried_actions)

        # A node is fully expanded if all possible actions have been tried
        # AND if there are no possible actions (meaning it's terminal or stuck)
        # For Tic-Tac-Toe, if no untried actions, it implies all children have been created.
        return not self.untried_actions

    def best_child(self, c_param: float = 1.4) -> 'MCTSNode':
        """
        Selects the best child node based on the UCB1 formula.
        The UCB1 formula needs `total_reward` and `visits`.

        Args:
            c_param (float): The exploration parameter (higher value encourages more exploration).

        Returns:
            MCTSNode: The selected child node.

        Raises:
            RuntimeError: If the node has no children to select from.
        """
        if not self.children:
            raise RuntimeError("Cannot select best child from a node with no children.")
        if self.visits == 0:  # Should not happen if a node has children and is selected
            # If parent visits are 0, log(0) is undefined. Handle by returning first child or raise error.
            # In typical MCTS, parent node is updated before children are selected for UCB1.
            return random.choice(list(self.children.values()))

        log_parent_visits = math.log(self.visits)  # This is N(p) in the UCB1 formula

        def ucb1_score(child_node: MCTSNode):
            if child_node.visits == 0:
                return float('inf')  # Prioritize unvisited children greatly

            # Exploitation Term: Average reward of the child
            exploitation_term = child_node.total_reward / child_node.visits  # Q(v) / N(v)

            # Exploration Term: Encourages visiting less-explored children
            exploration_term = c_param * math.sqrt(log_parent_visits / child_node.visits)  # C * sqrt(ln N(p) / N(v))

            return exploitation_term + exploration_term

        # Select the child with the highest UCB1 score
        return max(self.children.values(), key=ucb1_score)


    def update(self, reward: float):
        """
        Updates the visit count and total reward for this node.

        Args:
            reward (float): The reward obtained from a simulation passing through this node.
                            This reward is for the player whose turn it was *at the root of the simulation*.
        """
        self.visits += 1
        self.total_reward += reward

    def __repr__(self):
        """
        Returns a string representation of the MCTSNode object.
        """
        action_repr = f"({self.action[0]},{self.action[1]})" if self.action else "ROOT"
        avg_reward = self.total_reward / self.visits if self.visits > 0 else 0.0
        return (f"Node(Player={self.state.current_player_id}, Action={action_repr}, "
                f"Visits={self.visits}, AvgReward={avg_reward:.2f})")