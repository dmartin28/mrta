import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot
import phase1.generate_clusters_rand as gc
from phase1.generate_clusters_rand import generate_clusters_move
from phase1.generate_clusters_rand import generate_clusters_merge
from phase1.generate_clusters_rand import generate_clusters_movemerge
from phase1.generate_clusters_rand import nash_eq_clustering
from phase2.IP_assignment import IP_assignment
from phase2.IP_assignment_all_assigned import IP_assignment_all_assigned
import phase1.phase1_utils as utils

nu = 10 #number of robots
mu = 5 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task
L_t = 4 # Max number of tasks in a cluster
L_r = 4 # Max number of robots in a cluster
num_tests = 20

#Define the environment size
max_x = 50
min_x = 0
max_y = 50
min_y = 0

####### Define some specific task types: ############

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

#Define the two robot types:

robot_type_1 = [1,0]
robot_type_2 = [0,1]

total_clustered_reward = 0
total_optimal_reward = 0


for test in range(num_tests):
    print(f"\n--- Test {test + 1} ---")

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

    # Generate clusters
    print(robot_list)
    if len(robot_list) == 0 or len(task_list) == 0:
        print("Error, no robots or tasks to cluster")
        clusters = []
    else:
        clusters = gc.nash_eq_clustering(robot_list, task_list, L)
        clusters = gc.refine_clusters_merge(clusters, robot_list, task_list, L_r, L_t)

    # Perform the phase 2 optimal assignment inside each cluster
    cluster_assignments = []
    cluster_assign_rewards = []
    for i in range(len(clusters)):
        cluster = clusters[i]
        assignment, reward = IP_assignment([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L, kappa)
        cluster_assignments.append(assignment)
        cluster_assign_rewards.append(reward)

    clustered_reward = sum(cluster_assign_rewards)
    
    # Calculate optimal assignment
    optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, L, kappa)

    # Print results for this test
    print(f"Clustered Reward: {clustered_reward}")
    print(f"Clustered Assignments:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i}: Robots {cluster[0]}, Tasks {cluster[1]}")
        print(f"Assignment: {cluster_assignments[i]}")
        print(f"Reward: {cluster_assign_rewards[i]}")
    print(f"Optimal Reward: {optimal_reward}")

    # Update totals
    total_clustered_reward += clustered_reward
    total_optimal_reward += optimal_reward

# Calculate and print averages
avg_clustered_reward = total_clustered_reward / num_tests
avg_optimal_reward = total_optimal_reward / num_tests

print("\n--- Final Results ---")
print(f"Average Clustered Reward: {avg_clustered_reward}")
print(f"Average Optimal Reward: {avg_optimal_reward}")
print(f"Percent: {100*avg_clustered_reward/avg_optimal_reward}%")