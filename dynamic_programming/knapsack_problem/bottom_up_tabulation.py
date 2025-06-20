"""
0/1 Knapsack Problem - Bottom-Up Tabulation Approach
This implementation uses dynamic programming to solve the 0/1 Knapsack problem using a bottom-up approach with tabulation.
The 0/1 Knapsack problem is a classic optimization problem where we aim to maximize the total value of items that can fit into a knapsack of a given capacity, with the constraint that each item can either be included or excluded (hence "0/1").
"""

def knapsack_01_bottom_up(weights, values, capacity):
    n = len(weights)

    # dp[i][w] will store the maximum value for the first 'i' items with capacity 'w'
    # Initialize a (n+1) x (capacity+1) table with zeros
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill the dp table
    # i iterates through the items (from 1 to n)
    for i in range(1, n + 1):
        # w iterates through the capacities (from 1 to 'capacity')
        for w in range(1, capacity + 1):
            # Get the weight and value of the current item (item i is at index i-1)
            current_item_weight = weights[i - 1]
            current_item_value = values[i - 1]

            # Case 1: If the current item's weight is greater than the current capacity 'w'
            # We cannot include this item, so the max value is the same as without this item
            dp[i][w] = dp[i - 1][w]

            # Case 2: If the current item's weight is less than or equal to current capacity 'w'
            # We have a choice: include the item or not include it
            if current_item_weight <= w:
                # Option A: Don't include the current item
                value_without_current = dp[i - 1][w]

                # Option B: Include the current item
                # Add its value to the max value obtained from previous items with remaining capacity
                value_with_current = current_item_value + dp[i - 1][w - current_item_weight]

                # Take the maximum of the two options
                dp[i][w] = max(value_without_current, value_with_current)

    # The result is stored at dp[n][capacity]
    return dp[n][capacity]


# Test the bottom-up solution
weights1 = [1, 3, 4, 5]
values1 = [1, 4, 5, 7]
capacity1 = 7
print(
    f"Knapsack (bottom-up) for capacity {capacity1} with weights {weights1} and values {values1}: {knapsack_01_bottom_up(weights1, values1, capacity1)}")  # Expected: 9

weights2 = [10, 20, 30]
values2 = [60, 100, 120]
capacity2 = 50
print(
    f"Knapsack (bottom-up) for capacity {capacity2} with weights {weights2} and values {values2}: {knapsack_01_bottom_up(weights2, values2, capacity2)}")  # Expected: 220