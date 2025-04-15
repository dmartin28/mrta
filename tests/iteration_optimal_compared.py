"""
This file iteratively clusters and reassigns robots to tasks and compares with optimal assignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from phase2.IP_assignment import IP_assignment
from algorithms.cluster_assignment_rand import cluster_assignment_rand
import test_utils as tu

"""HyperParameters"""
nu = 9  # number of robots # was 10
mu = 5  # number of tasks  # was 5
kappa = 2  # number of capabilities
L = 3  # maximum team size for a single task
L_r = 5  # Max number of robots in a cluster
L_t = 5  # Max number of tasks in a cluster
num_iterations = 100  # number of iterations to run
num_tests = 10  # number of random tests to run

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
    'L_t': L_t,   # maximum number of tasks in a cluster
    'L_r': L_r,   # maximum number of robots in a cluster
}

# Define the environment size
max_x = 100
max_y = 100

# Run multiple tests and collect statistics
matches = 0
total_tests = 0
optimal_rewards = []
random_cluster_rewards = []

for test in range(num_tests):
    print(f"\n--- Test {test + 1}/{num_tests} ---")
    
    # Generate random scenario
    robot_list, task_list = tu.generate_problem_instance(hypes, max_x, max_y)
    
    # Run random clustering method
    total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(robot_list,task_list,num_iterations,hypes)
    
    # Get the final reward from the random clustering method
    random_final_reward = iteration_rewards[-1]
    
    # Calculate optimal assignment
    optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, hypes)

    # Compare results
    print(f"Optimal Reward: {optimal_reward}")
    print(f"Random Clustering Final Reward: {random_final_reward}")
    
    # Check if the random clustering method matches the optimal reward
    # Using a small epsilon for floating point comparison
    epsilon = 1e-6
    if abs(random_final_reward - optimal_reward) < epsilon:
        matches += 1
        print("MATCH!")
    else:
        print(f"Difference: {optimal_reward - random_final_reward}")
    
    total_tests += 1
    optimal_rewards.append(optimal_reward)
    random_cluster_rewards.append(random_final_reward)

# Calculate statistics
match_percentage = (matches / total_tests) * 100
avg_optimal_reward = np.mean(optimal_rewards)
avg_random_reward = np.mean(random_cluster_rewards)
avg_performance_ratio = np.mean(np.array(random_cluster_rewards) / np.array(optimal_rewards)) * 100

print("\n--- Final Statistics ---")
print(f"Total Tests: {total_tests}")
print(f"Total Matches: {matches}")
print(f"Match Percentage: {match_percentage:.2f}%")
print(f"Average Optimal Reward: {avg_optimal_reward:.2f}")
print(f"Average Random Clustering Reward: {avg_random_reward:.2f}")
print(f"Average Performance Ratio: {avg_performance_ratio:.2f}%")