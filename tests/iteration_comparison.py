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
nu = 50 #number of robots # was 10
mu = 50 # number of tasks  # was 5
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task

# Define the environment size
max_x = 100
min_x = 0
max_y = 100
min_y = 0

"Test Parameters"
cluster_sizes = [1,2,3,4,5,6,7]
num_tests = 50
num_iterations = 100 # number of iterations to run

# Initialize a dictionary to store results for each cluster size
results = {size: [] for size in cluster_sizes}

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
            # Perform random iterative assignment
            total_reward, iteration_assignments, iteration_rewards = cluster_assignment_rand(robot_list, task_list, cluster_size, cluster_size, kappa, num_iterations)
            
            # Store the iteration rewards for this test and cluster size
            results[cluster_size].append(iteration_rewards)

# Calculate average iteration rewards for each cluster size across all tests
avg_results = {size: np.mean(np.array(rewards), axis=0) for size, rewards in results.items()}

# Save results and avg_results to files
with open('random_cluster_results.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('random_cluster_avg_results.pkl', 'wb') as f:
    pickle.dump(avg_results, f)

print("Results saved to 'random_cluster_results.pkl' and 'random_cluster_avg_results.pkl'")

# Plot the results
plt.figure(figsize=(12, 8))
for size, avg_rewards in avg_results.items():
    # Create a new array with 0 at position 0, followed by the original rewards
    rewards_with_origin = np.insert(avg_rewards, 0, 0)
    # Plot starting from iteration 0
    plt.plot(range(0, num_iterations + 1), rewards_with_origin, label=f'Max Robots in Cluster: {size}')

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title(f'Average Reward vs. Iteration for Different Cluster Sizes\n(Averaged over {num_tests} tests with {nu} robots and {mu} tasks)')
plt.legend()
plt.grid(True)
plt.savefig('random_cluster_results_plot.png')
plt.show()