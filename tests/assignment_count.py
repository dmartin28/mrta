
"""
This script calculates the total number of ways to assign n robots to m tasks
Subject to the following constraints:
1. Each task can have at most L robots assigned to it.
2. Each robot can only be assigned to one task.

Note: We do not require all robots to be assigned to tasks.
Many or even all robots might be unassigned.
The first number in each partition represents the number of unassigned robots.
The remaining numbers represent the number of robots assigned to each task.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

# My Code:
import phase2.phase2_utils as utils

# Parameters:
n = 7  # number of robots
m = 7  # number of tasks
L = 3   # max number of robots per task

# Generate all partitions of n into m parts with each part <= L
partitions = utils.generate_partitions(n, m, L)

total_assignments = 0

# For each partition calculate the number of assignments using the multinomial coefficient
for partition in partitions:
    # Calculate the number of assignments for this partition
    # This is done using the multinomial coefficient
    # n! / (k1! * k2! * ... * km!)
    # where n is the total number of robots and ki is the number of robots in each partition
    partition_sum = sum(partition)
    numerator = math.factorial(partition_sum)
    denominator = 1
    for k in partition:
        denominator *= math.factorial(k)
    
    # Calculate the number of assignments for this partition
    num_assignments = numerator / denominator
    
    # Add number of assignments in this partition to the total
    total_assignments += num_assignments
    # Print the number of assignments for this partition
    print(f"Partition: {partition}, Number of Assignments: {num_assignments}")

# Print the total number of assignments
print(f"Total Number of Assignments for n={n}, m={m}: {int(total_assignments)}")