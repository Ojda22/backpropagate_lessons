"""
# Longest Common Subsequence (LCS) - Bottom-Up Tabulation Approach
# This implementation uses dynamic programming to find the length of the longest common subsequence
# between two strings using a bottom-up approach with tabulation.
# The LCS problem is a classic problem in computer science where we want to find the longest subsequence
# that appears in both strings in the same order, but not necessarily consecutively.
"""

def longest_common_subsequence_bottom_up(text1, text2):
    m = len(text1)
    n = len(text2)

    # dp[i][j] will store the LCS length of text1[0...i-1] and text2[0...j-1]
    # The table size needs to be (m+1) x (n+1) because dp[0][j] and dp[i][0] are base cases
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the dp table
    # i iterates through the rows (corresponding to text1 prefixes)
    for i in range(1, m + 1):
        # j iterates through the columns (corresponding to text2 prefixes)
        for j in range(1, n + 1):
            # If the current characters match
            if text1[i-1] == text2[j-1]:
                # Add 1 to the LCS of the previous prefixes
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                # If characters don't match, take the maximum of skipping from text1 or text2
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # The result is stored at dp[m][n], representing the full strings
    return dp[m][n]

# Test the bottom-up solution
text1_a, text2_a = "abcde", "ace"
print(f"LCS (bottom-up) of '{text1_a}' and '{text2_a}': {longest_common_subsequence_bottom_up(text1_a, text2_a)}") # Expected: 3

text1_b, text2_b = "abc", "abc"
print(f"LCS (bottom-up) of '{text1_b}' and '{text2_b}': {longest_common_subsequence_bottom_up(text1_b, text2_b)}") # Expected: 3

text1_c, text2_c = "abc", "def"
print(f"LCS (bottom-up) of '{text1_c}' and '{text2_c}': {longest_common_subsequence_bottom_up(text1_c, text2_c)}") # Expected: 0

text1_d, text2_d = "AGGTAB", "GXTXAYB"
print(f"LCS (bottom-up) of '{text1_d}' and '{text2_d}': {longest_common_subsequence_bottom_up(text1_d, text2_d)}") # Expected: 4