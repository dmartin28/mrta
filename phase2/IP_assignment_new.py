"""
This code takes a list of robots and tasks, and calculates the optimal
assignment of robots to tasks using a branch-and-bound approach.

The code first separates the search space into integer partitions that describe the
number of robots assigned to each task (but not which particular robots are assigned).
It then calculates upper and lower bounds for each partition. The partition with the 
highest upper bound is selected for further search. The process continues until all partitions 
have been evaluated or the upper bound of the remaining partitions is less than the global lower bound.

The best assignment is saved as a dictionary where the keys are global task IDs (including -1 for unassigned),
and the values are lists of robot IDs associated with each task.
"""

import numpy as np
import math
from itertools import combinations
from shared_classes.robot import Robot
import phase2.phase2_utils as utils

def IP_assignment_new(robot_list, task_list, L, kappa, printout=False):
    n = len(robot_list)
    m = len(task_list)

    best_assignment = None
    global_reward = float('-inf')

    # Check if there are no robots, then all tasks are unassigned
    if n == 0:
        reward = 0
        best_assignment = {-1: []}  # Start with empty list for unassigned robots
        for task_id in range(m):
            best_assignment[task_id] = []  # Empty list for each task
        return best_assignment, reward

    # Check if there are no tasks, then all robots are unassigned
    if m == 0:
        best_assignment = {-1: [robot.get_id() for robot in robot_list]}
        reward = 0
        return best_assignment, reward

    # Generate integer partitions of n into m+1 parts
    partitions = utils.generate_partitions(n, m, L)

    if printout:
        print(f"All partitions of {n} into {m}+1 parts with first part unassigned:")
        for partition in partitions:
            print(partition)
        print(f"Total number of partitions: {len(partitions)}")

    # Calculate mean and average reward for each possible team size:
    max_team_size = L
    max_rewards = np.full((max_team_size+1, m), float('-inf'))
    avg_rewards = np.zeros((max_team_size+1, m))

    # Calculate max and average reward for each team size for each task
    for team_size in range(0, max_team_size+1):
        for task_idx in range(0, m):
            combos = combinations(range(n), team_size)
            num_combos = math.comb(n, team_size)

            for combo in combos:                        
                team_capabilities = np.zeros(kappa, dtype=np.int32)
                for robot_idx in combo:
                    team_capabilities += robot_list[robot_idx].get_capabilities()

                # This gives the reward for completing a task with the given capabilities
                capability_value = task_list[task_idx].get_reward(*team_capabilities)
                
                # This is the cost of the team, assuming cost = distance traveled
                cost = sum(math.dist(robot_list[robot_idx].get_location(), task_list[task_idx].get_location()) for robot_idx in combo)
                
                net_reward = capability_value - cost
                
                if net_reward > max_rewards[team_size, task_idx]:
                    max_rewards[team_size, task_idx] = net_reward
                
                avg_rewards[team_size, task_idx] += net_reward/num_combos

    if printout:
        print("Average Rewards Matrix: Rows = Team Size, Columns = Task")
        print(avg_rewards)
        print("\nMaximum Rewards Matrix: Rows = Team Size, Columns = Task")
        print(max_rewards)

    # Calculate upper and lower bounds for each partition
    upper_bounds = np.zeros(len(partitions))
    lower_bounds = np.zeros(len(partitions))
    for partition_idx, partition in enumerate(partitions):
        for task_idx in range(m):
            lower_bounds[partition_idx] += avg_rewards[partition[task_idx+1], task_idx]
            upper_bounds[partition_idx] += max_rewards[partition[task_idx+1], task_idx]

    if printout:
        print("\nPartitions with Upper and Lower Bounds:")
        print("Partition | Lower Bound | Upper Bound")
        print("-" * 40)
        for partition_idx, partition in enumerate(partitions):
            print(f"{partition} | {lower_bounds[partition_idx]:.2f} | {upper_bounds[partition_idx]:.2f}")
        print(f"Total number of partitions: {len(partitions)}")

    # Find global lower bound
    LB = max(lower_bounds)

    # This loop searches partitions until all have been searched or pruned
    while len(partitions) > 0:
        best_partition_idx = np.argmax(upper_bounds)
        
        if printout:
            print("\nBest Partition:")
            print("Partition | Lower Bound | Upper Bound")
            print("-" * 40)
            print(f"{partitions[best_partition_idx]} | {lower_bounds[best_partition_idx]:.2f} | {upper_bounds[best_partition_idx]:.2f}")
        
        # Search partition with highest UB
        partition_assignment, partition_reward = utils.partition_search_dummyTask_new(robot_list, task_list, partitions[best_partition_idx])

        # Remove best partition from list
        partitions.pop(best_partition_idx)
        upper_bounds = np.delete(upper_bounds, best_partition_idx)

        if partition_reward > global_reward:
            global_reward = partition_reward
            best_assignment = partition_assignment
            LB = max(LB, global_reward)

        if printout:
            print("\nbest_assignment: ", best_assignment)
            print("global_reward: ", global_reward)
            print("LB: ", LB)

        # Trim all partitions whose UB is less than the LB
        for partition_idx in range(len(partitions) - 1, -1, -1):
            if upper_bounds[partition_idx] < LB:
                del partitions[partition_idx]
                upper_bounds = np.delete(upper_bounds, partition_idx)
                lower_bounds = np.delete(lower_bounds, partition_idx)

        if printout:
            print("\nTrimmed Partitions with Upper and Lower Bounds:")
            print("Partition | Lower Bound | Upper Bound")
            print("-" * 40)
            for partition_idx, partition in enumerate(partitions):
                print(f"{partition} | {lower_bounds[partition_idx]:.2f} | {upper_bounds[partition_idx]:.2f}")
            print("num partitions left: ", len(partitions))
            print("num_upper_bounds left: ", len(upper_bounds))

        # The best_assignment is already in the desired dictionary format, so we don't need to convert it

        if printout:
            print("Final best_assignment: ", best_assignment)
            print("Final global_reward: ", round(global_reward))

    return best_assignment, global_reward