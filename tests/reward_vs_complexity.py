"""
This file compares the results of the random merging method with different group size limits
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
from cluster_assignment_rand import cluster_assignment_rand

"""HyperParameters"""
nu = 30 #number of robots
mu = 30 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task

# Define the environment size
max_x = 100
min_x = 0
max_y = 100
min_y = 0

"Test Parameters"
cluster_sizes = [1, 2, 3, 4, 5, 6, 7]
num_tests = 50
test_time = 5  # Run each cluster size for 5 seconds

# Initialize dictionaries to store results and timing for each cluster size
reward_results = {size: [] for size in cluster_sizes}
time_results = {size: [] for size in cluster_sizes}
iterations_completed = {size: [] for size in cluster_sizes}

for test in range(num_tests):
    print(f"Test: {test+1}")

    """ Define some specific task types: """

    #Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
    reward_dim = tuple(L+1 for _ in range(kappa))

    #Type 0 can be done by robots with capability 1 
    task_type_0 = np.zeros(reward_dim)
    task_type_0[1,0] = 100
    task_type_0[2,0] = 200

    #Type 1 can be done by robots with capability 2 
    task_type_1 = np.zeros(reward_dim)
    task_type_1[0,1] = 100
    task_type_1[0,2] = 150
    task_type_1[0,3] = 200

    #Type 2 can be done only collaboratively by cap 1 and 2 
    task_type_2 = np.zeros(reward_dim)
    task_type_2[1,1] = 200
    task_type_2[1,2] = 250
    task_type_2[2,1] = 300

    #Type 3 can be done only collaboratively by two of cap 1 
    task_type_3 = np.zeros(reward_dim)
    task_type_3[2,0] = 200

    #Type 4 can be done only collaboratively by two of cap 2 
    task_type_4 = np.zeros(reward_dim)
    task_type_4[0,2] = 200

    #Type 5 can be done only collaboratively by cap 1 and 2 
    task_type_5 = np.zeros(reward_dim)
    task_type_5[1,1] = 200
    task_type_5[1,2] = 220
    task_type_5[2,1] = 220

    #Type 6 can be done only individually by type 1
    task_type_6 = np.zeros(reward_dim)
    task_type_6[1,0] = 100

    #Type 7 can be done only individually by type 2 
    task_type_7 = np.zeros(reward_dim)
    task_type_7[0,1] = 100

    #Type 8 can be done by either type, only individually:
    task_type_8 = np.zeros(reward_dim)
    task_type_8[1,0] = 100
    task_type_8[0,1] = 100

    #Type 9 can be done by either type but needs 3 robots:
    task_type_9 = np.zeros(reward_dim)
    task_type_9[1,2] = 350
    task_type_9[2,1] = 350
    task_type_9[3,0] = 350
    task_type_9[0,3] = 350
    ####################################################

    """ Define the two robot types: """

    robot_type_1 = [1,0] # Note this is the capability vector
    robot_type_2 = [0,1]

    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    robot_types = [robot_type_1, robot_type_2]
    for i in range(nu):
        robot_type = random.choice(robot_types)
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = [task_type_0, task_type_1, task_type_2, task_type_3, task_type_4,
                    task_type_5, task_type_6, task_type_7, task_type_8, task_type_9]
    for i in range(mu):
        task_type = random.choice(task_types)
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i])
        task_list.append(task)

    for cluster_size in cluster_sizes:
        # Set a large number of iterations as an upper limit
        max_iterations = 1000
        
        # Perform random iterative assignment with time limit
        total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
            robot_list, task_list, cluster_size, cluster_size, kappa, max_iterations, time_limit=test_time)
        
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