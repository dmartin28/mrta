import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

def assignment_count(n, m, L):
    memo = {}
    
    def f(n, m, L):
        if m == 0:
            return 1 if n == 0 else 0
        if n < 0:
            return 0
        
        if (n, m, L) in memo:
            return memo[(n, m, L)]
        
        total = 0
        if m == 1:
            total = 1
        else:
            for k in range(min(L, n) + 1):
                total += binomial_coefficient(n, k) * f(n-k, m-1, L)
        
        memo[(n, m, L)] = total
        return total
    
    return f(n, m+1, L)

# Fixed L value
L = 3

# Create a 2D list to store results
results = [[0 for _ in range(10)] for _ in range(8)]

# Calculate results for all combinations of n and m
for m in range(1, 9):
    for n in range(1, 11):
        results[m-1][n-1] = assignment_count(n, m, L)

# Print the results in a table format
print(f"Results for L = {L}:")
print("\nn\\m", end="")
for m in range(1, 9):
    print(f"{m:10d}", end="")
print("\n" + "-" * 90)

for n in range(1, 11):
    print(f"{n:2d} |", end="")
    for m in range(1, 9):
        print(f"{results[m-1][n-1]:10d}", end="")
    print()

print("\nNote: m represents the number of tasks (excluding the dummy task for unassigned robots)")
print("      n represents the number of robots")