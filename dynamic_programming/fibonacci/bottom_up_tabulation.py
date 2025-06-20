"""
Fibonacci using Bottom-Up Tabulation Approach
This implementation uses dynamic programming to compute Fibonacci numbers efficiently.
More specifically, it uses a bottom-up approach with tabulation to store intermediate results.
Significantly faster than the naive recursive approach.
"""
def fib_bottom_up(n):
    if n <= 1: # Base case: return n for 0 or 1
        return n # for recursion to work correctly, the very first thing is to make sure we have a base case

    # Create a table (array) to store the results of sub-problems
    # dp[i] will store the i-th Fibonacci number
    dp = [0] * (n + 1) # Initialize a list of size n+1 with zeros. N + 1 is used to include the 0th index, which corresponds to the Fibonacci number for 0.

    # Base cases
    dp[0] = 0
    dp[1] = 1

    # Fill the table iteratively
    # Each dp[i] is computed using previously computed values (dp[i-1] and dp[i-2])
    for i in range(2, n + 1): # Start from 2 because we already know dp[0] and dp[1]
        dp[i] = dp[i - 1] + dp[i - 2] # The Fibonacci relation: F(n) = F(n-1) + F(n-2)

    return dp[n] # Return the n-th Fibonacci number, which is stored in dp[n]

# Test the bottom-up solution
print(f"Fibonacci (bottom-up) of 6: {fib_bottom_up(6)}") # Expected: 8
print(f"Fibonacci (bottom-up) of 10: {fib_bottom_up(10)}") # Expected: 55

import time
start_time = time.time()
print(f"Fibonacci (bottom up) of 40: {fib_bottom_up(40)}") # Computes instantly
end_time = time.time()
print(f"Time taken for bottom up fib_bottom_up(40): {end_time - start_time} seconds") # Expected: instantaneous, much faster than the naive recursive approach