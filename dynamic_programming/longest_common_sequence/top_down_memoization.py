"""
# Longest Common Subsequence (LCS) using Top-Down Dynamic Programming with Memoization
# This implementation uses dynamic programming to find the length of the longest common subsequence
# between two strings using a top-down approach with memoization.
# The LCS problem is a classic problem in computer science where we want to find the longest subsequence
# that appears in both strings in the same order, but not necessarily consecutively.
"""

def longest_common_subsequence_top_down(text1, text2):
    m = len(text1)
    n = len(text2)

    # memo[i][j] will store the LCS length of text1[0...i-1] and text2[0...j-1]
    # Initialize with -1 to indicate not computed
    memo = {} # Using a dictionary for memoization, keys will be tuples (i, j)

    def solve(i, j):
        # Base cases: If either string is empty, LCS is 0
        if i == 0 or j == 0:
            return 0

        # Check if the result is already computed
        if (i, j) in memo:
            return memo[(i, j)]

        # If the last characters match
        if text1[i-1] == text2[j-1]:
            # The last character is part of the LCS, so add 1 and solve for remaining prefixes
            result = 1 + solve(i - 1, j - 1)
        else:
            # If characters don't match, take the maximum of skipping a character from text1
            # or skipping a character from text2
            result = max(solve(i - 1, j), solve(i, j - 1))

        # Store the result in memo before returning
        memo[(i, j)] = result
        return result

    # Start the recursion from the full strings
    return solve(m, n)

# Test the top-down solution
text1_a, text2_a = "abcde", "ace"
print(f"LCS (top-down) of '{text1_a}' and '{text2_a}': {longest_common_subsequence_top_down(text1_a, text2_a)}") # Expected: 3

text1_b, text2_b = "abc", "abc"
print(f"LCS (top-down) of '{text1_b}' and '{text2_b}': {longest_common_subsequence_top_down(text1_b, text2_b)}") # Expected: 3

text1_c, text2_c = "abc", "def"
print(f"LCS (top-down) of '{text1_c}' and '{text2_c}': {longest_common_subsequence_top_down(text1_c, text2_c)}") # Expected: 0

text1_d, text2_d = "AGGTAB", "GXTXAYB"
print(f"LCS (top-down) of '{text1_d}' and '{text2_d}': {longest_common_subsequence_top_down(text1_d, text2_d)}") # Expected: 4 ("GTAB")