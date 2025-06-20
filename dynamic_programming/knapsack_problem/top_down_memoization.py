
"""
0/1 Knapsack Problem - Bottom-Up Tabulation Approach
This implementation uses dynamic programming to solve the 0/1 Knapsack problem using a bottom-up approach with tabulation.
The 0/1 Knapsack problem is a classic optimization problem where we aim to maximize the total value of items that can fit into a knapsack of a given capacity, with the constraint that each item can either be included or excluded (hence "0/1").
"""
def knapsack_01_top_down(weights, values, capacity):
    n = len(weights)
    # memo[(i, w)] will store the max value using first 'i' items with capacity 'w'
    memo = {}

    def solve(i, current_capacity):
        # Base cases:
        # If no items left or no capacity left, max value is 0
        if i == 0 or current_capacity == 0:
            return 0

        # Check if result is already computed
        if (i, current_capacity) in memo:
            return memo[(i, current_capacity)]

        # If the weight of the current item (weights[i-1]) is greater than current_capacity,
        # we cannot include this item. So, we move to the next item (i-1) with the same capacity.
        if weights[i-1] > current_capacity:
            result = solve(i - 1, current_capacity)
        else:
            # We have two choices for the current item (weights[i-1], values[i-1]):
            # 1. Don't include the current item:
            value_without_current = solve(i - 1, current_capacity)

            # 2. Include the current item:
            # Add its value and solve for remaining items with reduced capacity
            value_with_current = values[i-1] + solve(i - 1, current_capacity - weights[i-1])

            # Take the maximum of the two choices
            result = max(value_without_current, value_with_current)

        # Store the result in memo before returning
        memo[(i, current_capacity)] = result
        return result

    # Start recursion from all 'n' items and the given 'capacity'
    return solve(n, capacity)

# Test the top-down solution
weights1 = [1, 3, 4, 5]
values1 = [1, 4, 5, 7]
capacity1 = 7
print(f"Knapsack (top-down) for capacity {capacity1} with weights {weights1} and values {values1}: {knapsack_01_top_down(weights1, values1, capacity1)}") # Expected: 9

weights2 = [10, 20, 30]
values2 = [60, 100, 120]
capacity2 = 50
print(f"Knapsack (top-down) for capacity {capacity2} with weights {weights2} and values {values2}: {knapsack_01_top_down(weights2, values2, capacity2)}") # Expected: 220 (20+30)