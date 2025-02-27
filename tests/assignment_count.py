import math

def binomial_coefficient(n, k):
    return math.comb(n, k)

def assignment_count(n, m, L):
    # Create a memoization dictionary to store computed results
    memo = {}
    
    def f(n, m, L):
        # Base cases
        if n < m:
            return 0
        if m == 1:
            return 1 if n <= L else 0
        
        # Check if result is already memoized
        if (n, m, L) in memo:
            return memo[(n, m, L)]
        
        # Recursive calculation
        total = 0
        for k in range(1, min(L, n-m+1) + 1):
            total += binomial_coefficient(n, k) * f(n-k, m-1, L)
        
        # Memoize the result
        memo[(n, m, L)] = total
        return total
    
    return f(n, m, L)

# Example usage
n = 10  # number of agents
m = 3   # number of subsets
L = 5   # maximum subset size

result = assignment_count(n, m, L)
print(f"Number of ways to partition {n} agents into {m} subsets with max size {L}: {result}")

# Interactive input
while True:
    try:
        n = int(input("Enter number of agents (n): "))
        m = int(input("Enter number of subsets (m): "))
        L = int(input("Enter maximum subset size (L): "))
        
        result = assignment_count(n, m, L)
        print(f"Number of ways to partition {n} agents into {m} subsets with max size {L}: {result}")
    except ValueError:
        print("Please enter valid integers.")
    except KeyboardInterrupt:
        print("\nExiting the program.")
        break
    print()  # Add a blank line for readability