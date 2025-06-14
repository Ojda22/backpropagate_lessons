import random

import numpy as np
import math
import time

# --- 1. Grid World Configuration ---

# Define the grid layout
# 'S': Start
# 'G': Goal (positive reward)
# 'X': Obstacle/Pitfall (negative reward)
# ' ': Empty cell (small negative reward to encourage movement)
GRID = [
    [' ', ' ', ' ', 'G'],
    [' ', 'X', ' ', ' '],
    [' ', ' ', ' ', ' '],
    ['S', ' ', ' ', ' ']
]

# Get grid dimensions
GRID_ROWS = len(GRID)
GRID_COLS = len(GRID[0])

# Define actions
ACTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

# Mapping for policy visualization
ACTION_SYMBOLS = {
    'UP': '↑',
    'DOWN': '↓',
    'LEFT': '←',
    'RIGHT': '→'
}

# --- 2. MDP Parameters ---

# Rewards
REWARD_EMPTY = -0.04  # Small negative reward for each step to encourage faster paths
REWARD_GOAL = 1.0  # Positive reward for reaching the goal
REWARD_PITFALL = -1.0  # Negative reward for falling into a pitfall

# Discount Factor (gamma)
# Determines the importance of future rewards.
# Closer to 1: future rewards are very important.
# Closer to 0: agent focuses on immediate rewards.
GAMMA = 0.9

# Transition Probabilities (for an action like 'UP')
# 80% chance of intended move, 10% chance of slipping left, 10% chance of slipping right
P_INTENDED = 0.8
P_SLIP_SIDE = 0.1  # Probability of slipping to the side
P_STAY = 0.0  # Probability of staying in place (not used here, but could be)


# --- 3. Helper Functions ---

def get_state_coords(state_idx):
    """Converts a flat state index to (row, col) coordinates."""
    return (state_idx // GRID_COLS, state_idx % GRID_COLS)


def get_state_idx(row, col):
    """Converts (row, col) coordinates to a flat state index."""
    return row * GRID_COLS + col


def is_valid_state(row, col):
    """Checks if a given (row, col) is within grid boundaries."""
    return 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS


def get_reward(row, col):
    """Returns the reward for entering a specific cell."""
    cell_type = GRID[row][col]
    if cell_type == 'G':
        return REWARD_GOAL
    elif cell_type == 'X':
        return REWARD_PITFALL
    else:
        return REWARD_EMPTY


def get_possible_next_states_and_probs(s_row, s_col, action) -> dict[tuple[int, int], float]:
    """
    Calculates the possible next states and their probabilities
    given a current state (s_row, s_col) and an action.
    Considers slipping.
    """
    next_states_probs: dict[tuple[int, int], float] = {}  # {(next_row, next_col): probability}

    # Intended move
    dr, dc = ACTIONS[action]
    next_row_intended, next_col_intended = s_row + dr, s_col + dc

    # Slipping moves (perpendicular to intended)
    # If action is UP/DOWN, slipping is LEFT/RIGHT
    # If action is LEFT/RIGHT, slipping is UP/DOWN
    if action in ['UP', 'DOWN']:
        dr_slip1, dc_slip1 = ACTIONS['LEFT']
        dr_slip2, dc_slip2 = ACTIONS['RIGHT']
    else:  # LEFT, RIGHT
        dr_slip1, dc_slip1 = ACTIONS['UP']
        dr_slip2, dc_slip2 = ACTIONS['DOWN']

    next_row_slip1, next_col_slip1 = s_row + dr_slip1, s_col + dc_slip1
    next_row_slip2, next_col_slip2 = s_row + dr_slip2, s_col + dc_slip2

    # Collect valid next states and their probabilities

    # Intended move
    if is_valid_state(next_row_intended, next_col_intended) and GRID[next_row_intended][next_col_intended] != 'X':
        next_states_probs[(next_row_intended, next_col_intended)] = next_states_probs.get(
            (next_row_intended, next_col_intended), 0) + P_INTENDED
    else:
        # If intended move hits a wall or obstacle, agent stays in current state
        next_states_probs[(s_row, s_col)] = next_states_probs.get((s_row, s_col), 0) + P_INTENDED

    # Slip 1
    if is_valid_state(next_row_slip1, next_col_slip1) and GRID[next_row_slip1][next_col_slip1] != 'X':
        next_states_probs[(next_row_slip1, next_col_slip1)] = next_states_probs.get((next_row_slip1, next_col_slip1),
                                                                                    0) + P_SLIP_SIDE
    else:
        next_states_probs[(s_row, s_col)] = next_states_probs.get((s_row, s_col), 0) + P_SLIP_SIDE

    # Slip 2
    if is_valid_state(next_row_slip2, next_col_slip2) and GRID[next_row_slip2][next_col_slip2] != 'X':
        next_states_probs[(next_row_slip2, next_col_slip2)] = next_states_probs.get((next_row_slip2, next_col_slip2),
                                                                                    0) + P_SLIP_SIDE
    else:
        next_states_probs[(s_row, s_col)] = next_states_probs.get((s_row, s_col), 0) + P_SLIP_SIDE

    # return probabilities of states of slipping (unintended moves) based on action taken and current state
    # return will always be a dictionary with keys as (row, col) tuples and values as probabilities
    # it move is valid it will return intended move with 80% probability and two unintended moves with 10% each
    # if the intended move is invalid (e.g. hitting a wall or obstacle), it will return the current state with 80% probability
    # and the two unintended moves with 10% each, if they are valid, otherwise unintended moves will also return current state with 80% probability
    # if the intended move is valid, and both unintended moves are also valid, the probabilities will be 100% distributed as 80% for intended and 10% each for unintended moves
    return next_states_probs


# --- 4. Value Iteration Algorithm ---

def value_iteration(theta=1e-4, max_iterations=1000):
    """
    Performs Value Iteration to find the optimal value function V*(s).

    Args:
        theta (float): A small threshold for convergence.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.

    Returns:
        V (numpy.ndarray): The converged optimal value function for each state.
    """

    # Initialize values for all states to 0
    # V[row, col] = estimated value of that state
    V = np.zeros((GRID_ROWS, GRID_COLS))

    print("\n--- Starting Value Iteration ---")
    print(f"Grid size: {GRID_ROWS}x{GRID_COLS}")
    print(f"Discount Factor (gamma): {GAMMA}")
    print(f"Action probabilities: Intended={P_INTENDED}, Slip Side={P_SLIP_SIDE}")
    print(f"Convergence Threshold (theta): {theta}\n")

    for iteration in range(max_iterations): # In each iteration refine the values in V
        delta = 0  # Stores the maximum change in value during this iteration

        V_new = np.copy(V)  # Create a copy to store updated values for this iteration

        print(f"Iteration {iteration + 1} of {max_iterations}... ")
        print_values(V_new)

        for r in range(GRID_ROWS):  # Iterate over each row
            for c in range(GRID_COLS): # Iterate over each column ( inner loop for each cell in the grid)
                # Terminal states (Goal and Pitfall) have fixed values, do not update
                if GRID[r][c] == 'G' or GRID[r][c] == 'X':
                    continue # Skip terminal states, they have their own fixed values, we are looking for optimal values of non-terminal states
                    # the fixed value is already set in the initialization of V to 0, which is fine for terminal states

                v_s = V[r, c]  # Current value of the state. First iteration will be zero, but it will be updated in the next iterations

                q_values = {} # Calculate Q-values for each action (expected value of taking action 'a' in state 's')
                for action_name, _ in ACTIONS.items(): # Iterate over each action in the grid
                    expected_value_for_action = 0  # Initialize expected value for this action

                    # Get all possible next states and their probabilities for this action given the current state
                    next_states_and_probs: dict[tuple[int, int], float] = get_possible_next_states_and_probs(r, c, action_name)

                    # Sum up expected values from all possible next states.
                    # Next states are the states that can be reached from the current state by taking the action
                    # Plus, unexpected states that can be reached by slipping (unintended moves)
                    for (next_r, next_c), prob in next_states_and_probs.items():
                        reward_s_prime = get_reward(next_r, next_c) # Get the immediate reward for the next state as defined in the grid
                        value_s_prime = V[next_r, next_c] # Get the current estimated value of the next state
                        # values_s_prime is the value of the next state, which is already estimated in V. First iteration will be zero, but it will be updated in the next iterations
                        # P(s'|s,a) - transition probability (the probability of reaching the next state s' from the current state s by taking action a)
                        # R(s') - immediate reward for the next state s'
                        # V(s') - estimated value of the next state s' (calculated over iterations)
                        # gamma - discount factor (how much we care about future rewards)
                        expected_value_for_action += prob * (reward_s_prime + GAMMA * value_s_prime) # Q(s,a) = sum(P(s'|s,a) * (R(s') + gamma * V(s')))

                    q_values[action_name] = expected_value_for_action # We store all expected values for each action from the current state

                # After calculating Q-values for all actions, we can update the value of the state
                # Update the value of the state to the maximum Q-value (the best expected value of taking specific action from the current state)
                # V(s) = max_a Q(s,a)
                V_new[r, c] = max(q_values.values()) if q_values else v_s  # Handle no Q-values (e.g. terminal)

                # Calculate the change for convergence check
                delta = max(delta, abs(V_new[r, c] - v_s))

        V = V_new  # Update V for the next iteration

        # Print progress to visualize search
        if iteration % 10 == 0 or delta < theta:
            print(f"Iteration {iteration:4d}: Max change (delta) = {delta:.6f}")
            # print_values(V, f"Values after Iteration {iteration}") # Uncomment for more detailed step-by-step

        # Check for convergence
        if delta < theta:
            print(f"\nValue Iteration converged after {iteration} iterations (delta < {theta}).")
            break

    else:  # If loop completes without break
        print(f"\nValue Iteration reached max_iterations ({max_iterations}) without converging (delta = {delta:.6f}).")

    return V


# --- 5. Policy Extraction ---

def extract_policy(V):
    """
    Derives the optimal policy from the optimal value function V*(s).
    """
    policy = np.empty((GRID_ROWS, GRID_COLS), dtype='U1')  # Use Unicode string for symbols

    print("\n--- Extracting Optimal Policy ---")

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if GRID[r][c] in ['G', 'X']:
                policy[r, c] = GRID[r][c]  # Terminal states keep their identity
                continue

            best_action_value = -math.inf
            best_action = None

            # we look for the best action from each state based on Q-value for each action
            for action_name, _ in ACTIONS.items():
                expected_value_for_action = 0
                next_states_and_probs = get_possible_next_states_and_probs(r, c, action_name)

                for (next_r, next_c), prob in next_states_and_probs.items():
                    reward_s_prime = get_reward(next_r, next_c)
                    value_s_prime = V[next_r, next_c]
                    expected_value_for_action += prob * (reward_s_prime + GAMMA * value_s_prime)

                if expected_value_for_action > best_action_value:
                    best_action_value = expected_value_for_action
                    best_action = action_name

            policy[r, c] = ACTION_SYMBOLS[best_action] if best_action else '?'  # '?' if no best action found

    return policy


# --- 6. Visualization Functions ---

def print_grid_layout():
    """Prints the initial grid layout."""
    print("--- Grid Layout ---")
    for r in range(GRID_ROWS):
        print(" ".join(GRID[r]))
    print("\nLegend: S=Start, G=Goal, X=Pitfall, ' '=Empty\n")


def print_values(V, title="Optimal State Values (V*(s))"):
    """Prints the state values in a formatted grid."""
    print(f"\n--- {title} ---")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            print(f"{V[r, c]:7.3f}", end=" ")  # Format to 3 decimal places
        print()  # New line after each row
    print()


def print_policy(policy, title="Optimal Policy ($\pi$*(s))"):
    """Prints the optimal policy in a formatted grid."""
    print(f"\n--- {title} ---")
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            print(f"  {policy[r, c]}  ", end=" ")
        print()  # New line after each row
    print()


# --- Main Execution ---

if __name__ == "__main__":
    print_grid_layout() # we print the initial grid layout to help us understand better the policy

    start_time = time.time() # we start at this time

    # 1. Run Value Iteration to find optimal values
    optimal_values = value_iteration() # this will return the optimal values for each state in the grid

    end_time = time.time() # we end at this time
    print(f"Value Iteration took {end_time - start_time:.4f} seconds.")

    # 2. Print the final optimal state values
    print_values(optimal_values)

    # 3. Extract and print the optimal policy
    optimal_policy = extract_policy(optimal_values) # this will return the optimal policy for each state in the grid
    print_policy(optimal_policy)

    print("\n--- Simulation of an Optimal Path (Demonstration) ---")
    # Find the start state
    start_r, start_c = -1, -1
    for r_idx in range(GRID_ROWS):
        for c_idx in range(GRID_COLS):
            if GRID[r_idx][c_idx] == 'S':
                start_r, start_c = r_idx, c_idx
                break
        if start_r != -1:
            break

    if start_r == -1:
        print("Error: Start state 'S' not found in the grid!")
    else:
        current_r, current_c = start_r, start_c
        path = [(current_r, current_c)]
        total_reward = 0
        max_steps = 50  # Prevent infinite loops in case of errors

        print(f"Starting at ({current_r}, {current_c})")

        # Simulate an agent following the optimal policy (with randomness)
        for step in range(max_steps):
            if GRID[current_r][current_c] in ['G', 'X']:
                print(f"Reached terminal state: {GRID[current_r][current_c]} at ({current_r}, {current_c}).")
                break

            chosen_action_symbol = optimal_policy[current_r, current_c]
            chosen_action_name = None
            for name, symbol in ACTION_SYMBOLS.items():
                if symbol == chosen_action_symbol:
                    chosen_action_name = name
                    break

            if chosen_action_name is None:  # Should not happen if policy is well-formed
                print(f"No valid action for state ({current_r}, {current_c}).")
                break

            print(
                f"  Step {step + 1}: Current state ({current_r}, {current_c}), Policy suggests action: {chosen_action_name} ({chosen_action_symbol})")

            # Get possible next states with their probabilities based on the chosen action
            possible_next_states_and_probs = get_possible_next_states_and_probs(current_r, current_c,
                                                                                chosen_action_name)

            # Select the actual next state probabilistically
            next_state_candidates = list(possible_next_states_and_probs.keys())
            probabilities = list(possible_next_states_and_probs.values())

            # Normalize probabilities if they don't sum to 1.0 due to floating point
            sum_probs = sum(probabilities)
            if sum_probs != 0:
                probabilities = [p / sum_probs for p in probabilities]

            if not next_state_candidates:
                print("No next states found from current state.")
                break

            next_r, next_c = random.choices(next_state_candidates, weights=probabilities, k=1)[0]

            immediate_reward = get_reward(next_r, next_c)
            total_reward += immediate_reward

            print(
                f"    Actual transition (due to probability): To ({next_r}, {next_c}) with reward {immediate_reward:.2f}")

            current_r, current_c = next_r, next_c
            path.append((current_r, current_c))

        print(f"\nPath taken: {path}")
        print(f"Total reward accumulated: {total_reward:.2f}")
        print("Note: The path taken is probabilistic, so it may not always follow the 'straightest' path due to slips.")