"""
This file iteratively clusters and reassigns robots to tasks and compares with optimal assignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot
import phase1.generate_clusters as gc
from phase2.IP_assignment import IP_assignment
from phase1.convert_assignment_to_clusters import convert_assignment_to_clusters
from cluster_assignment_rand import cluster_assignment_rand
import copy

"""HyperParameters"""
nu = 10  # number of robots # was 10
mu = 10  # number of tasks  # was 5
kappa = 2  # number of capabilities
L = 3  # maximum team size for a single task
L_r = 7  # Max number of robots in a cluster
L_t = 7  # Max number of tasks in a cluster
num_iterations = 100  # number of iterations to run
num_tests = 100  # number of random tests to run

# Define the environment size
max_x = 100
min_x = 0
max_y = 100
min_y = 0

def create_task_types():
    """Define some specific task types"""
    # Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
    reward_dim = tuple(L+1 for _ in range(kappa))

    # Type 0 can be done by robots with capability 1 
    task_type_0 = np.zeros(reward_dim)
    task_type_0[1,0] = 100
    task_type_0[2,0] = 200

    # Type 1 can be done by robots with capability 2 
    task_type_1 = np.zeros(reward_dim)
    task_type_1[0,1] = 100
    task_type_1[0,2] = 150
    task_type_1[0,3] = 200

    # Type 2 can be done only collaboratively by cap 1 and 2 
    task_type_2 = np.zeros(reward_dim)
    task_type_2[1,1] = 200
    task_type_2[1,2] = 250
    task_type_2[2,1] = 300

    # Type 3 can be done only collaboratively by two of cap 1 
    task_type_3 = np.zeros(reward_dim)
    task_type_3[2,0] = 200

    # Type 4 can be done only collaboratively by two of cap 2 
    task_type_4 = np.zeros(reward_dim)
    task_type_4[0,2] = 200

    # Type 5 can be done only collaboratively by cap 1 and 2 
    task_type_5 = np.zeros(reward_dim)
    task_type_5[1,1] = 200
    task_type_5[1,2] = 220
    task_type_5[2,1] = 220

    # Type 6 can be done only individually by type 1
    task_type_6 = np.zeros(reward_dim)
    task_type_6[1,0] = 100

    # Type 7 can be done only individually by type 2 
    task_type_7 = np.zeros(reward_dim)
    task_type_7[0,1] = 100

    # Type 8 can be done by either type, only individually:
    task_type_8 = np.zeros(reward_dim)
    task_type_8[1,0] = 100
    task_type_8[0,1] = 100

    # Type 9 can be done by either type but needs 3 robots:
    task_type_9 = np.zeros(reward_dim)
    task_type_9[1,2] = 350
    task_type_9[2,1] = 350
    task_type_9[3,0] = 350
    task_type_9[0,3] = 350
    
    return [task_type_0, task_type_1, task_type_2, task_type_3, task_type_4,
            task_type_5, task_type_6, task_type_7, task_type_8, task_type_9]

def create_robot_types():
    """Define the two robot types"""
    robot_type_1 = [1,0]  # Note this is the capability vector
    robot_type_2 = [0,1]
    return [robot_type_1, robot_type_2]

def generate_random_scenario():
    """Generate random robots and tasks"""
    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    robot_types = create_robot_types()
    for i in range(nu):
        robot_type = random.choice(robot_types)
        robot = Robot(i, robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = create_task_types()
    for i in range(mu):
        task_type = random.choice(task_types)
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i])
        task_list.append(task)
        
    return robot_list, task_list

# Run multiple tests and collect statistics
matches = 0
total_tests = 0
optimal_rewards = []
random_cluster_rewards = []

for test in range(num_tests):
    print(f"\n--- Test {test + 1}/{num_tests} ---")
    
    # Generate random scenario
    robot_list, task_list = generate_random_scenario()
    
    # Run random clustering method
    total_reward, iteration_assignments, iteration_rewards = cluster_assignment_rand(
        robot_list, task_list, L_r, L_t, kappa, num_iterations, printout=False)
    
    # Get the final reward from the random clustering method
    random_final_reward = iteration_rewards[-1]
    
    # Calculate optimal assignment
    optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, L, kappa)
    
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