
import math

def stirling_number_second_kind(n, m):
    memo = {}
    
    def stirling(n, m):
        if n == m:
            return 1
        if m == 0 or m > n:
            return 0
        if m == 1 or m == n:
            return 1
        
        if (n, m) in memo:
            return memo[(n, m)]
        
        result = m * stirling(n-1, m) + stirling(n-1, m-1)
        memo[(n, m)] = result
        return result
    
    return stirling(n, m)

def total_assignments(n, m):
    return stirling_number_second_kind(n, m) * math.factorial(m)

# Calculate and print total assignments for n = 1 to 10 and m = 1 to 8
print("Total number of ways to assign n robots to m tasks:")
print("\nn\\m", end="")
for m in range(1, 9):
    print(f"{m:12d}", end="")
print("\n" + "-" * 108)

for n in range(1, 11):
    print(f"{n:2d} |", end="")
    for m in range(1, 9):
        if m > n:
            print(f"{0:12d}", end="")
        else:
            print(f"{total_assignments(n, m):12d}", end="")
    print()

print("\nNote: n represents the number of robots")
print("      m represents the number of tasks")
print("This table shows the total number of ways to assign n robots to m tasks,")
print("calculated as: Stirling number of the second kind * m!")