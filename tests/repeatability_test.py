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
nu = 50  # number of robots
mu = 30  # number of tasks
kappa = 2  # number of capabilities
L = 3  # maximum team size for a single task

# Define the environment size
max_x, min_x = 100, 0
max_y, min_y = 100, 0

"Test Parameters"
cluster_size = 6
num_runs = 10
num_iterations = 1000  # number of iterations to run

# Initialize a list to store results for each run
results = []

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

for run in range(num_runs):
    print(f"Run: {run+1}")

    # Perform random iterative assignment
    total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
        robot_list, task_list, cluster_size, cluster_size, kappa, num_iterations
    )
    
    # Store the iteration rewards for this run
    results.append(iteration_rewards)

# Save results to file
with open('random_cluster_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to 'random_cluster_results.pkl'")

# Plot the results
plt.figure(figsize=(12, 8))
for run, rewards in enumerate(results):
    # Create a new array with 0 at position 0, followed by the original rewards
    rewards_with_origin = np.insert(rewards, 0, 0)
    # Plot starting from iteration 0
    plt.plot(range(0, num_iterations + 1), rewards_with_origin, label=f'Run {run+1}')

plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title(f'Reward vs. Iteration for 5 Runs\n(Cluster Size: {cluster_size}, {nu} robots and {mu} tasks)')
plt.legend()
plt.grid(True)
plt.savefig('repeatability_test.png')
plt.show()