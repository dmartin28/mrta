"""This function generates an assignment by creating clusters of robots and tasks using a neural network as a guide.
Inputs:
        num_iterations = number of clustering iterations to perform
        L_r = maximum number of robots in a cluster
        L_t = maximum number of tasks in a cluster
        kappa = number of different robot capabilities
        
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
import phase1.generate_clusters as gc
from phase2.IP_assignment import IP_assignment
from phase1.convert_assignment_to_clusters import convert_assignment_to_clusters
import copy
import time


def cluster_assignment_nn(robot_list, task_list, L_r, L_t, kappa, num_iterations, printout=False):
    # Initialize the rewards, assignment vectors, and time tracking
    iteration_rewards = []
    iteration_assignments = []
    iteration_times = []  # New list to track time per iteration

    # Create empty assignment_groupings list
    assignment_groupings = []

    #print(f"num_iterations: {num_iterations}")
    for iteration in range(num_iterations):
        # Initialize timing variables
        total_iteration_time = 0
        
        """ 1. Start with all robots and tasks in their own individual assignment grouping """
        # Note: Assignment groupings have same shape as clusters [List of robots, List of tasks] (2D array)
        if iteration == 0:
            assignment_groupings = []  # Reset assignment_groupings for first iteration
            for robot in robot_list:
                assignment_groupings.append([[robot.id], []])
            for task in task_list:
                assignment_groupings.append([[], [task.id]])

        if printout:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        """ 2. Merge assignment groupings to create clusters """
        # Merge assignment groupings to create clusters - NOT timed
        clusters = gc.refine_clusters_random_merge(assignment_groupings, L_r, L_t)

        # Start timing after cluster creation
        start_time = time.time()

        """3. Perform optimal assignment within each cluster"""
        cluster_assignments = []
        cluster_assign_rewards = []
        for cluster in clusters:
            
            # Perform optimal assignment
            
            # L is max robots per task
            L = len(task_list[0].get_reward_matrix())-1
            
            assignment, reward = IP_assignment([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L, kappa)
            
            # Store cluster assignments and rewards
            cluster_assignments.append(assignment)
            cluster_assign_rewards.append(reward)
        
        """ Convert the cluster assignments to a single assignment """
        # initialize the assignment list
        num_tasks = len(task_list) # total number of tasks
        assignment = [[] for _ in range(num_tasks + 1)]  # Stores the global assignment

        for cluster_idx, cluster in enumerate(clusters):
            cluster_assignment = cluster_assignments[cluster_idx]
            
            # Add unassigned robots to the assignment
            assignment[0].extend(cluster_assignment[0])
            
            # Add assigned robots to their respective tasks
            for task_idx, task_id in enumerate(cluster[1]):
                if task_idx + 1 < len(cluster_assignment):  # Check if the task has an assignment
                    assignment[task_id + 1].extend(cluster_assignment[task_idx + 1])

        """ 4. Create assignment groupings based on the current assignment """
        assignment_groupings = convert_assignment_to_clusters(assignment)

        # Output the results of the current iteration:
        total_reward = sum(cluster_assign_rewards)
        if printout:
            print(f"Total Reward: {total_reward}")
            print(f"Assignment: {assignment}")

        # Store the results of the current iteration
        iteration_rewards.append(total_reward)
        iteration_assignments.append(copy.deepcopy(assignment))
        
        # Calculate and store the time taken for this iteration (excluding cluster creation)
        end_time = time.time()
        iteration_time = end_time - start_time
        iteration_times.append(iteration_time)
        
    return total_reward, iteration_assignments, iteration_rewards, iteration_times