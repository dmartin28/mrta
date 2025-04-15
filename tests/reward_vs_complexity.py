"""
This file compares the results of the random merging method with different group size limits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot
import matplotlib.pyplot as plt
import pickle
from algorithms.cluster_assignment_rand import cluster_assignment_rand
import test_utils as tu

"""HyperParameters"""
nu = 30 #number of robots
mu = 30 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task

# Define the environment size
max_x = 100
max_y = 100

"Test Parameters"
cluster_sizes = [1, 2, 3, 4, 5, 6, 7]
num_tests = 1
test_time = 5  # Run each cluster size for 5 seconds

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
}

# Initialize dictionaries to store results and timing for each cluster size
reward_results = {size: [] for size in cluster_sizes}
time_results = {size: [] for size in cluster_sizes}
iterations_completed = {size: [] for size in cluster_sizes}

for test in range(num_tests):
    print(f"Test: {test+1}")

    # Generate Problem Instance
    robot_list, task_list = tu.generate_problem_instance(hypes, max_x, max_y)

    for cluster_size in cluster_sizes:
        # Set a large number of iterations as an upper limit
        max_iterations = 1000
        
        # Update hyperparameters for the current cluster size
        hypes['L_r'] = cluster_size  # maximum number of robots in a cluster
        hypes['L_t'] = cluster_size  # maximum number of tasks in a cluster

        # Perform random iterative assignment with time limit
        total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
            robot_list, task_list, max_iterations, hypes, time_limit=test_time)
        
        # Calculate total execution time
        total_time = sum(iteration_times)
        
        # Store the iteration rewards and times for this test and cluster size
        reward_results[cluster_size].append(iteration_rewards)
        time_results[cluster_size].append(iteration_times)
        iterations_completed[cluster_size].append(len(iteration_rewards))
        
        print(f"Cluster size {cluster_size}: {total_time:.2f} seconds total, {len(iteration_rewards)} iterations, {total_time/len(iteration_rewards):.6f} seconds per iteration")

# Calculate average iteration rewards and times for each cluster size across all tests
avg_results = {size: [] for size in cluster_sizes}
avg_times = {size: [] for size in cluster_sizes}

# Find the maximum number of iterations completed for any cluster size
max_iterations_across_all = max([max([len(rewards) for rewards in reward_results[size]]) for size in cluster_sizes])

# Process the results to create arrays of equal length for plotting
for size in cluster_sizes:
    # Find the maximum number of iterations for this cluster size across all tests
    max_iter_for_size = max([len(rewards) for rewards in reward_results[size]])
    
    # For each test, pad the rewards and times arrays to the max_iter_for_size
    padded_rewards = []
    padded_times = []
    
    for test_idx in range(num_tests):
        rewards = reward_results[size][test_idx]
        times = time_results[size][test_idx]
        
        # Pad rewards with the last value
        if len(rewards) < max_iter_for_size:
            padded_rewards.append(np.pad(rewards, (0, max_iter_for_size - len(rewards)), 'edge'))
        else:
            padded_rewards.append(rewards)
            
        # Pad times with zeros (we'll only use actual times for cumulative time calculation)
        if len(times) < max_iter_for_size:
            padded_times.append(np.pad(times, (0, max_iter_for_size - len(times)), 'constant'))
        else:
            padded_times.append(times)
    
    # Calculate average rewards and times across all tests
    avg_results[size] = np.mean(padded_rewards, axis=0)
    avg_times[size] = np.mean(padded_times, axis=0)

# After all tests are completed, calculate and print average time per iteration for each cluster size
print("\nAverage time per iteration across all tests:")
for cluster_size in cluster_sizes:
    # Calculate average time per iteration across all tests for this cluster size
    all_iteration_times = []
    for test_times in time_results[cluster_size]:
        all_iteration_times.extend(test_times)
    
    avg_time_per_iteration = sum(all_iteration_times) / len(all_iteration_times)
    avg_iterations = np.mean(iterations_completed[cluster_size])
    
    print(f"Cluster size {cluster_size}: {avg_time_per_iteration:.6f} seconds per iteration, avg {avg_iterations:.1f} iterations completed")

# Save results and avg_results to files
with open('random_cluster_results.pkl', 'wb') as f:
    pickle.dump(reward_results, f)

with open('random_cluster_avg_results.pkl', 'wb') as f:
    pickle.dump(avg_results, f)

with open('random_cluster_time_results.pkl', 'wb') as f:
    pickle.dump(time_results, f)

with open('random_cluster_avg_times.pkl', 'wb') as f:
    pickle.dump(avg_times, f)

print("Results saved to pickle files")

# Plot 1: Reward vs Iteration
plt.figure(figsize=(12, 8))
for size, avg_rewards in avg_results.items():
    # Create a new array with 0 at position 0, followed by the original rewards
    rewards_with_origin = np.insert(avg_rewards, 0, 0)
    # Plot starting from iteration 0
    plt.plot(range(0, len(rewards_with_origin)), rewards_with_origin, label=f'Max Robots in Cluster: {size}')

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title(f'Average Reward vs. Iteration for Different Cluster Sizes\n(Averaged over {num_tests} tests with {nu} robots and {mu} tasks, {test_time}s time limit)')
plt.legend()
plt.grid(True)
plt.savefig('random_cluster_results_plot.png')
plt.show()

# Plot 2: Reward vs Elapsed Time (limited to test_time)
plt.figure(figsize=(12, 8))

# Create a common time grid for all cluster sizes from 0 to test_time
common_time_grid = np.linspace(0, test_time, 100)

for size in cluster_sizes:
    # For each cluster size, we need to process each test separately
    all_interpolated_rewards = []
    
    for test_idx in range(num_tests):
        rewards = reward_results[size][test_idx]
        times = time_results[size][test_idx]
        
        # Calculate cumulative times
        cumulative_times = np.cumsum(times)
        
        # Add a point at (0,0) for the start
        cumulative_times = np.insert(cumulative_times, 0, 0)
        rewards_with_zero = np.insert(rewards, 0, 0)
        
        # Use numpy's interp function for linear interpolation
        # This will limit the interpolation to the range of cumulative_times
        # For times beyond the last cumulative time, use the last reward value
        interp_rewards = np.interp(
            common_time_grid, 
            cumulative_times, 
            rewards_with_zero,
            right=rewards_with_zero[-1]  # Use the last reward for extrapolation
        )
        
        all_interpolated_rewards.append(interp_rewards)
    
    # Average the interpolated rewards across all tests
    avg_interpolated_rewards = np.mean(all_interpolated_rewards, axis=0)
    
    # Plot the averaged result
    plt.plot(common_time_grid, avg_interpolated_rewards, label=f'Max Robots in Cluster: {size}')

plt.xlabel('Elapsed Time (seconds)')
plt.ylabel('Average Reward')
plt.xlim(0, test_time)  # Limit x-axis to range from 0 to test_time
plt.title(f'Average Reward vs. Elapsed Time for Different Cluster Sizes\n(Averaged over {num_tests} tests with {nu} robots and {mu} tasks, {test_time}s time limit)')
plt.legend()
plt.grid(True)
plt.savefig('random_cluster_time_plot.png')
plt.show()