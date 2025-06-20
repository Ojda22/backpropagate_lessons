"""
Naive Fibonacci implementation using recursion.
This implementation is inefficient for larger values of n due to exponential time complexity.
"""
def fib_recursive(n):
    if n <= 1: # Base case: return n for 0 or 1
        return n # for recursion to work correctly, the very first think is to make sure we have a base case
    else:
        return fib_recursive(n - 1) + fib_recursive(n - 2) # Recursive case: sum of the two preceding numbers, subtracting 1 and 2 from n. Creating subsequences of the problem until we reach the base case.

# Test the recursive solution
print(f"Fibonacci (recursive) of 6: {fib_recursive(7)}") # for 6, we calculate
print(f"Fibonacci (recursive) of 10: {fib_recursive(8)}") # Expected: 55

import time
start_time = time.time()
print(f"Fibonacci (recursive) of 40: {fib_recursive(40)}") # Computes but takes a long time
end_time = time.time()
print(f"Time taken for naive fib_recursive(40): {end_time - start_time} seconds") # Expected: a long time