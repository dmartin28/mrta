import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shared_classes.task import Task
from shared_classes.robot import Robot
from phase1.generate_clusters import generate_clusters
from phase2.IP_assignment import IP_assignment
from phase2.IP_assignment_all_assigned import IP_assignment_all_assigned
import phase1.phase1_utils as utils

nu = 0 #number of robots
mu = 0 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task
L_t = 10 # Max number of tasks in a cluster
L_r = 8 # Max number of robots in a cluster

#Define the environment size
max_x = 50
min_x = 0
max_y = 50
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

"""---Start Generate Clusters---"""

clusters = generate_clusters(robot_list,task_list,L_r,L_t)

"""---End Generate Clusters---"""

# Perform the phase 2 optimal assignment inside each cluster:
cluster_assignments = []
cluster_assign_rewards = []
for i in range(len(clusters)):
    cluster = clusters[i]
    assignment, reward = IP_assignment([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L,kappa)
    cluster_assignments.append(assignment)
    cluster_assign_rewards.append(reward)

print("Final clusters: ")
print()
for i in range(len(clusters)):
    cluster = clusters[i]
    print("Cluster: ", i)
    print("Robots: ", cluster[0])
    print("Tasks: ", cluster[1])
    #print("Cluster Value: ", utils.coalition_value_1([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], kappa))
    print("Cluster Value: ", utils.coalition_value_2([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], kappa))
    # Print cluster capabilities
    cluster_capabilities = np.zeros(kappa, dtype=int)
    for robot in cluster[0]:
        cluster_capabilities += robot_list[robot].capabilities.astype(int)
    print("Robot Capabilities: ", cluster_capabilities)
    
    print("Assignment: ", cluster_assignments[i])
    print("Net Reward: ", cluster_assign_rewards[i])
    print()

clustered_reward = sum(cluster_assign_rewards)

# This is somewhat confusing, we unscramble the clustering assignments
# Task id is the index in the original task list
# Task idx is the index in the cluster
# and match tasks/assignments to their original i.d.

#Global assignment of robots to tasks
global_assignment = [[] for _ in range(mu)]
unused_robots = []

for i in range(len(cluster_assignments)):
    #print()
    #print("assignments[", i, "]: ", cluster_assignments[i])
    
    # Add first group of robots to unused_robots
    for robot_id in cluster_assignments[i][0]:
        unused_robots.append(robot_id)
    
    # Add assignment for each task to the global assignment
    for task_idx in range(0,len(clusters[i][1])): #for each task in a cluster i
        #print("i: ", i , " Task idx: ", task_idx)
        #print("Cluster:", clusters[i])
        task_id = (clusters[i][1][task_idx]) # [cluster][0/1=robots/tasks][task or robot idx]
        global_assignment[task_id] = cluster_assignments[i][task_idx+1]

print("unused_robots: ", unused_robots)
global_assignment.insert(0,unused_robots)    

#Compare to direct optimal assignment:
print("Clustered Reward: ", sum(cluster_assign_rewards))
print("Clustered Assignment: ", global_assignment)
optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, L, kappa)
print("Optimal Assignment: ", optimal_assignment)
print("Optimal Reward: ", optimal_reward)