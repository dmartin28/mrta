import numpy as np
from task import Task
from robot import Robot
from IP_assignment import IP_assignment
import phase1_utils as utils

nu = 11 #number of robots
mu = 4 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task
L_t = 4 # Max number of tasks in a cluster
L_r = 8 # Max number of robots in a cluster

#Define the environment size
max_x = 10
min_x = 0
max_y = 10
min_y = 0

####### Define some specific task types: ############

#Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
reward_dim = tuple(L+1 for _ in range(kappa))

#Type 1 can be done by robots with capability 1 
task_type_0 = np.zeros(reward_dim)
task_type_0[1,0] = 100
task_type_0[2,0] = 200

#Type 2 can be done by robots with capability 2 
task_type_1 = np.zeros(reward_dim)
task_type_1[0,1] = 100
task_type_1[0,2] = 150
task_type_1[0,3] = 200

#Type 3 can be done only collaboratively by cap 1 and 2 
task_type_2 = np.zeros(reward_dim)
task_type_2[1,1] = 200
task_type_2[1,2] = 250
task_type_2[2,1] = 300

#Type 4 can be done only collaboratively by two of cap 1 
task_type_3 = np.zeros(reward_dim)
task_type_3[2,0] = 200

#Type 5 can be done only collaboratively by two of cap 2 
task_type_4 = np.zeros(reward_dim)
task_type_4[0,2] = 200

# #Type 6 can be done only collaboratively by cap 1 and 2 
# task_type_6 = np.zeros(reward_dim)
# task_type_6[1,1] = 200
# task_type_6[1,2] = 220
# task_type_6[2,1] = 220

# #Type 7 can be done only collaboratively by two of cap 1 
# task_type_7 = np.zeros(reward_dim)
# task_type_7[1,0] = 100

# #Type 8 can be done only collaboratively by two of cap 2 
# task_type_8 = np.zeros(reward_dim)
# task_type_8[0,1] = 100
#####################################################

#Define the two robot types:

robot_type_1 = [1,0]
robot_type_2 = [0,1]

# Generate random robot and task locations
robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

robot_list = []
task_list = []

# Create robots
for i in range(nu):
    if i % 2 == 0:
        robot_type = robot_type_1
    else:
        robot_type = robot_type_2
    
    robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
    robot_list.append(robot)

# Create tasks
task_types = [task_type_0, task_type_1, task_type_2, task_type_3, task_type_4] 
for i in range(mu):
    task = Task(i, task_types[i%5], task_x_locations[i], task_y_locations[i])
    task_list.append(task)

# Check the robots created
print(f"Created {len(robot_list)} robots:")
for robot in robot_list:
    print(f"Robot {robot.id}: Position ({robot.x}, {robot.y})")
    print(f"Capabilities: {robot.capabilities}")
    print()  # Add an empty line for better readability

# Check the tasks created
print(f"\nCreated {len(task_list)} tasks:")
for task in task_list:
    print(f"Task {task.id}: Position ({task.x}, {task.y})")
    print(f"Reward Matrix:\n{task.reward_matrix}")
    print()  # Add an empty line for better readability

# Create a clustering of tasks and robots
# cluster1_robots = [robot_list[0], robot_list[1]]
# cluster1_tasks = [task_list[0],task_list[2]]

# cluster1_robots = [robot_list[0], robot_list[1], robot_list[2]]
# cluster1_tasks = [task_list[0],task_list[1],task_list[2],task_list[3]]

# print("Cluster 1:")
# for robot in cluster1_robots:
#     print(f"Robot {robot.id}: Position ({robot.x}, {robot.y})")
#     print(f"Capabilities: {robot.capabilities}")
#     print()
# for task in cluster1_tasks:
#     print(f"Task {task.id}: Position ({task.x}, {task.y})")
#     print(f"Reward Matrix:\n{task.reward_matrix}")
#     print("Grand Coalition: ", task.get_grand_coalition())
#     print()

# Calculate the value of the cluster:
# c1_value = utils.coalition_value_1(cluster1_robots, cluster1_tasks, kappa)

"""Initialize each robot + task in their individual cluster"""
clusters = []
for robot in robot_list:
    clusters.append([[robot.id], []])
for task in task_list:
    clusters.append([[], [task.id]])
# Each cluster is 2D array, row 1 are robot indices, row 2 are task indices

# For each pair of clusters, calculate the value of the merged cluster
equilibrium = False
while equilibrium == False:
    max_change = 0
    merge_indices = []
    for i in range(len(clusters)):
        for j in range (i+1, len(clusters)):
            
            # print("Checking clusters: ", clusters[i], clusters[j])
            # print("Robot indices: ", clusters[i][0], clusters[j][0])
            # print("Task indices: ", clusters[i][1], clusters[j][1])
            # print("clusters[i][0]) + len(clusters[j][0])", len(clusters[i][0]) + len(clusters[j][0]))
            # print("clusters[i][1]) + len(clusters[j][1])", len(clusters[i][1]) + len(clusters[j][1]))
            # Check if the merge would exceed the maximum cluster sizes
            if len(clusters[i][0]) + len(clusters[j][0]) < L_r and len(clusters[i][1]) + len(clusters[j][1]) < L_t:
                merged_cluster = [clusters[i][0] + clusters[j][0], clusters[i][1] + clusters[j][1]]
                #print("Merged cluster: ", merged_cluster)
                merged_value = utils.coalition_value_1([robot_list[r] for r in merged_cluster[0]], [task_list[t] for t in merged_cluster[1]], kappa)

                #Can probably store these values so we don't recompute each time:
                clusteri_val = utils.coalition_value_1([robot_list[r] for r in clusters[i][0]], [task_list[t] for t in clusters[i][1]], kappa)
                clusterj_val = utils.coalition_value_1([robot_list[r] for r in clusters[j][0]], [task_list[t] for t in clusters[j][1]], kappa)
                difference = merged_value - clusteri_val - clusterj_val

                if difference > max_change:
                    max_change = difference
                    merge_indices = [i,j]

    if max_change > 0:
        # Merge the clusters
        clusters[merge_indices[0]][0] += clusters[merge_indices[1]][0]
        clusters[merge_indices[0]][1] += clusters[merge_indices[1]][1]
        clusters.pop(merge_indices[1])
    else:
        equilibrium = True

# Perform the phase 2 optimal assignment inside each cluster:
assignments = []
rewards = []
for i in range(len(clusters)):
    cluster = clusters[i]
    assignment, reward = IP_assignment([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L,kappa)
    assignments.append(assignment)
    rewards.append(reward)

print("Final clusters: ")
for i in range(len(clusters)):
    cluster = clusters[i]
    print("Cluster: ", i)
    print("Robots: ", cluster[0])
    print("Tasks: ", cluster[1])
    print("Cluster Value: ", utils.coalition_value_1([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], kappa))
    
    # Print cluster capabilities
    cluster_capabilities = np.zeros(kappa, dtype=int)
    for robot in cluster[0]:
        cluster_capabilities += robot_list[robot].capabilities.astype(int)
    print("Robot Capabilities: ", cluster_capabilities)
    
    print("Assignment: ", assignments[i])
    print("Net Reward: ", rewards[i])
    print()

clustered_reward = sum(rewards)

# This is somewhat confusing, we unscramble the clustering assignments
# Task id is the index in the original task list
# Task idx is the index in the cluster
# and match tasks/assignments to their original i.d.
clustered_assignment = [[] for _ in range(mu)]
for i in range(len(assignments)):
    for task_idx in range(len(assignments[i])):
        task_id = (clusters[i][1][task_idx])
        clustered_assignment[task_id] = assignments[i][task_idx]

#Compare to direct optimal assignment:
optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, L, kappa)
print("Optimal Assignment: ", optimal_assignment)
print("Optimal Reward: ", optimal_reward)
print("Clustered Reward: ", sum(rewards))
print("Clustered Assignment: ", clustered_assignment)

# Generate a clustering of tasks and robots

# Initialize clusters
# Each cluster will contain list of robot indices and task indices
# Each cluster begins with one robot or one task

# Calculate the reward for each cluster
# Reward should be 0 if the cluster contains only one robot or one task

# Clustering approach 1: Merge two clusters that will provide greatest increase in reward
# Basic structure of the clustering algorithm
# While the algorithm has not converged:
#     Search the neighborhood of the current clustering for the best merge
#     Merge the clusters
#     Update the reward for the new cluster
#     Repeat until convergence

# Clustering approach 2: Treat as a nash equilibrium seeking clustering
# For each robot, calculate the marginal contribution to each cluster
    # Join the coalition to which the robot has the highest marginal contribution
# For each task, calculate the marginal contribution to each cluster



# Assignment, Reward = IP_assignment(robot_list, task_list, L)
# print("Assignment: ", Assignment, "Reward: ", Reward)