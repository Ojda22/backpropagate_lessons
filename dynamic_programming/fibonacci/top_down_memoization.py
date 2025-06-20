"""
Top-Down Dynamic Programming Approach to Fibonacci Sequence with Memoization
This implementation uses memoization to store previously computed Fibonacci numbers,
significantly improving performance over the naive recursive approach.
"""
def fib_top_down(n, memo={}):
    if n <= 1: # Base case: return n for 0 or 1
        return n # for recursion to work correctly, the very first think is to make sure we have a base case

    # Check if the result is already computed (memoized)
    if n in memo:
        return memo[n] # Return the memoized result if it exists

    # If not computed, compute and store the result
    result = fib_top_down(n - 1, memo) + fib_top_down(n - 2, memo)
    memo[n] = result
    return result

# Test the top-down solution
# Important: When calling multiple times, you might want to reset the memo for independent calls
# or pass an empty dictionary for each new top-level call.
print(f"Fibonacci (top-down) of 6: {fib_top_down(6, {})}") # Expected: 8
print(f"Fibonacci (top-down) of 10: {fib_top_down(10, {})}") # Expected: 55


import time
start_time = time.time()
print(f"Fibonacci (top-down) of 40: {fib_top_down(40)}") # Computes instantly
end_time = time.time()
print(f"Time taken for top down fib_top_down(40): {end_time - start_time} seconds") # Expected: instantaneous, much faster than the naive recursive approach