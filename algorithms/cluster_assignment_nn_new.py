"""This function generates an assignment by creating clusters of robots and tasks using a neural network as a guide.
Inputs:
        num_iterations = number of clustering iterations to perform
        L_r = maximum number of robots in a cluster
        L_t = maximum number of tasks in a cluster
        kappa = number of different robot capabilities
        
The Algorithm is as follows:
1. Start with all robots and tasks in their own individual assignment grouping
2. For each iteration:
    1. Merge assignment groupings to create clusters (Using Neural Network)
    2. Perform optimal assignment within each cluster
    3. Calculate and output the total reward of the current assignment
    4. Create assignment groupings based on the current assignment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.IP_assignment_new import IP_assignment_new
from phase1.convert_assignment_to_clusters import convert_assignment_to_clusters_new
from phase1.generate_clusters_nn import refine_clusters_nn_merge
import copy
import time

def cluster_assignment_nn_new(model, robot_list, task_list, L_r, L_t, kappa, num_iterations, epsilon=0.1, printout=False):
    # Initialize the rewards, assignment vectors, and time tracking
    iteration_rewards = []
    iteration_assignments = []
    iteration_times = []  # New list to track time per iteration

    # Create empty assignment_groupings list
    assignment_groupings = []

    # L is max robots per task
    L = len(task_list[0].get_reward_matrix())-1

    for iteration in range(num_iterations):
        # Initialize timing variables
        total_iteration_time = 0
        
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
        clusters = refine_clusters_nn_merge(assignment_groupings, robot_list, task_list, L_r, L_t, kappa, L, model, epsilon)

        # Start timing after cluster creation
        start_time = time.time()

        """3. Perform optimal assignment within each cluster"""
        cluster_assignments = []
        cluster_assign_rewards = []
        for cluster in clusters:
            # Perform optimal assignment 
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
        
    return total_reward, iteration_assignments, iteration_rewards, iteration_times