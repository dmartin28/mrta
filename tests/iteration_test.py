"""
This file iteratively clusters and reassigns robots to tasks

The Algorithm is as follows:
1. Start with all robots and tasks in their own individual assignment grouping
2. For each iteration:
    1. Merge assignment groupings to create clusters
    2. Perform optimal assignment within each cluster
    3. Calculate and output the total reward of the current assignment
    4. Create assignment groupings based on the current assignment
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
nu = 9 #number of robots # was 10
mu = 5 # number of tasks  # was 5
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task
L_t = 7 # Max number of tasks in a cluster
L_r = 7 # Max number of robots in a cluster
num_iterations = 100 # number of iterations to run

# Define the environment size
max_x = 50
min_x = 0
max_y = 50
min_y = 0

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

# #Type 9 can be done by either type, and collaboratively:
# task_type_9 = np.zeros(reward_dim)
# task_type_9[1,0] = 100
# task_type_9[0,1] = 100
# task_type_9[1,1] = 250

#Type 9 can be done by either type but needs 3 robots:
#Note the Shapley value calculation will not really understand this
# since there is no clear maximum value
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

# # Initialize the rewards and assignment vecctors
# iteration_rewards = []
# iteration_assignments = []

# """ 1. Start with all robots and tasks in their own individual assignment grouping """
# # Note: Assignment groupings have same shape as clusters [List of robots, List of tasks] (2D array)
# assignment_groupings = []
# for robot in robot_list:
#     assignment_groupings.append([[robot.id], []])
# for task in task_list:
#     assignment_groupings.append([[], [task.id]])

# for iteration in range(num_iterations):
#     print(f"\n--- Iteration {iteration + 1} ---")
    
#     """ 2. Merge assignment groupings to create clusters """
#     #clusters = gc.refine_clusters_merge(assignment_groupings,robot_list, task_list, L_r, L_t)
#     clusters = gc.refine_clusters_random_merge(assignment_groupings, L_r, L_t)

#     """3. Perform optimal assignment within each cluster"""
#     cluster_assignments = []
#     cluster_assign_rewards = []
#     for cluster in clusters:
        
#         # Perform optimal assignment
#         assignment, reward = IP_assignment([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L, kappa)
        
#         # Store cluster assignments and rewards
#         cluster_assignments.append(assignment)
#         cluster_assign_rewards.append(reward)
    
#     """ Convert the cluster assignments to a single assignment """

#     """ Convert the cluster assignments to a single assignment """
#     # initialize the assignment list
#     num_tasks = mu # total number of tasks
#     assignment = [[] for _ in range(num_tasks + 1)]  # Stores the global assignment

#     for cluster_idx, cluster in enumerate(clusters):
#         cluster_assignment = cluster_assignments[cluster_idx]
        
#         # Add unassigned robots to the assignment
#         assignment[0].extend(cluster_assignment[0])
        
#         # Add assigned robots to their respective tasks
#         for task_idx, task_id in enumerate(cluster[1]):
#             if task_idx + 1 < len(cluster_assignment):  # Check if the task has an assignment
#                 assignment[task_id + 1].extend(cluster_assignment[task_idx + 1])

#     """ 4. Create assignment groupings based on the current assignment """
#     assignment_groupings = convert_assignment_to_clusters(assignment)

#     # # Print cluster assignments, assignment, and assignment_groupings:
#     # for cluster_assignment_idx in range(len(cluster_assignments)):
#     #     print(f"Cluster {cluster_assignment_idx} Tasks: {clusters[cluster_assignment_idx][1]}")
#     #     print(f"Cluster {cluster_assignment_idx} Robots: {clusters[cluster_assignment_idx][0]}")
#     #     print(f"Cluster {cluster_assignment_idx} Assignments: {cluster_assignments[cluster_assignment_idx]}")
#     #     print("\n")    
#     # print(f"Assignment: {assignment}")
#     # print(f"Assignment Groupings: {assignment_groupings}")

#     # Output the results of the current iteration:
#     total_reward = sum(cluster_assign_rewards)
#     print(f"Total Reward: {total_reward}")
#     print(f"Assignment: {assignment}")

#     # Store the results of the current iteration
#     iteration_rewards.append(total_reward)
#     iteration_assignments.append(copy.deepcopy(assignment))

total_reward, iteration_assignments, iteration_rewards = cluster_assignment_rand(robot_list, task_list, L_r, L_t, kappa, num_iterations)

# Print final results of all iterations
print("\n--- Final Results ---")
for i in range(len(iteration_rewards)):
    #print(f"Iteration {i + 1}:")
    #print(f"Total Reward: {iteration_rewards[i]}")
    #print(f"Assignment: {iteration_assignments[i]}")
    print(f"Iteration {i + 1}: Total Reward: {iteration_rewards[i]}")

# Calculate optimal assignment
optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, L, kappa)
print(f"\nOptimal Reward: {optimal_reward}")
print(f"Optimal Assignment: {optimal_assignment}")
print(f"Iterative Assignment: {iteration_assignments[-1]}")