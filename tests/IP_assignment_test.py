# Phase 2 assignment algorithm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shared_classes.task import Task
from phase2.IP_assignment_all_assigned import IP_assignment_all_assigned
from phase2.IP_assignment import IP_assignment
from phase2.IP_assignment_new import IP_assignment_new
from phase2.exhaustive_search import exhaustive_search
from shared_classes.robot import Robot
import tests.test_utils as utils

""" Hyper Parameters"""
nu = 8  # number of robots
mu = 5  # number of tasks
L = 3  # maximum team size
kappa = 2  # number of capabilities

# Define a dictionary to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
}

#Define the environment size
max_x = 10
min_x = 0
max_y = 10
min_y = 0

task_types = utils.generate_task_types(L, kappa)

robot_list, task_list = utils.generate_problem_instance(hypes, max_x, max_y)

utils.print_problem_instance(robot_list, task_list)

# # Run the algorithms
# Assignment_old, Reward_old = IP_assignment(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)
# Assignment_new, Reward_new = IP_assignment_new(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)
# exhaustive_assignment, ex_reward = exhaustive_search(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)

# Print results
print("\nResults Comparison:")
print("-" * 50)
print(f"{'Algorithm':<20} {'Reward':<10} {'Assignment'}")
print("-" * 50)
Assignment_old, Reward_old = IP_assignment(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)
print(f"{'IP_assignment':<20} {Reward_old:<10.2f} {Assignment_old}")
Assignment_new, Reward_new = IP_assignment_new(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)
print(f"{'IP_assignment_new':<20} {Reward_new:<10.2f} {Assignment_new}")
#exhaustive_assignment, ex_reward = exhaustive_search(robot_list, task_list, hypes['L'], hypes['kappa'], printout=False)
#print(f"{'Exhaustive Search':<20} {ex_reward:<10.2f} {exhaustive_assignment}")
print("-" * 50)