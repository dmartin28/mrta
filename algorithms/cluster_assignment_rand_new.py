"""This function generates an assignment by creating random clusters of robots and tasks
Inputs:
        robot_list = list of robot objects
        task_list = list of task objects
        L_r = maximum number of robots in a cluster
        L_t = maximum number of tasks in a cluster
        kappa = number of different robot capabilities
        num_iterations = maximum number of clustering iterations to perform
        time_limit = maximum execution time in seconds (optional)
        printout = whether to print progress (optional)
        
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

import phase1.generate_clusters_rand as gc
from phase2.IP_assignment import IP_assignment_new
from phase1.convert_assignment_to_clusters import convert_assignment_to_clusters_new
import copy
import time

def cluster_assignment_rand_new(robot_list, task_list, L_r, L_t, kappa, num_iterations, time_limit=None, printout=False):
    # Initialize the rewards, assignment vectors, and time tracking
    iteration_rewards = []
    iteration_assignments = []
    iteration_times = []  # List to track time per iteration
    
    total_time_elapsed = 0  # Track total time elapsed
    
    # Create empty assignment_groupings list
    assignment_groupings = []

    for iteration in range(num_iterations):
        # Check if we've exceeded the time limit
        if time_limit is not None and total_time_elapsed >= time_limit:
            if printout:
                print(f"Time limit of {time_limit} seconds reached after {iteration} iterations.")
            break
            
        """ 1. Start with all robots and tasks in their own individual assignment grouping """
        if iteration == 0:
            assignment_groupings = []  # Reset assignment_groupings for first iteration
            for robot in robot_list:
                assignment_groupings.append([[robot.id], []])
            for task in task_list:
                assignment_groupings.append([[], [task.id]])

        if printout:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        """ 2. Merge assignment groupings to create clusters """
        clusters = gc.refine_clusters_random_merge(assignment_groupings, L_r, L_t)

        # Start timing after cluster creation
        start_time = time.time()

        """3. Perform optimal assignment within each cluster"""
        cluster_assignments = []
        cluster_assign_rewards = []
        for cluster in clusters:
            # L is max robots per task
            L = len(task_list[0].get_reward_matrix())-1
            
            assignment, reward = IP_assignment_new([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], L, kappa)
            
            # Store cluster assignments and rewards
            cluster_assignments.append(assignment)
            cluster_assign_rewards.append(reward)
        
        """ Convert the cluster assignments to a single assignment """
        # Initialize the global assignment dictionary
        assignment = {-1: []}  # Start with empty list for unassigned robots
        for task in task_list:
            assignment[task.id] = []  # Empty list for each task

        for cluster_idx, cluster in enumerate(clusters):
            cluster_assignment = cluster_assignments[cluster_idx]
            
            # Add unassigned robots to the global assignment
            assignment[-1].extend(cluster_assignment.get(-1, []))
            
            # Add assigned robots to their respective tasks in the global assignment
            for task_id in cluster[1]:
                if task_id in cluster_assignment:
                    assignment[task_id].extend(cluster_assignment[task_id])

        """ 4. Create assignment groupings based on the current assignment """
        assignment_groupings = convert_assignment_to_clusters_new(assignment)

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
        
        # Update total time elapsed
        total_time_elapsed += iteration_time
        
        # Check if we're about to exceed the time limit
        if time_limit is not None and total_time_elapsed >= time_limit:
            if printout:
                print(f"Time limit of {time_limit} seconds reached after {iteration + 1} iterations.")
            break
        
    return total_reward, iteration_assignments, iteration_rewards, iteration_times