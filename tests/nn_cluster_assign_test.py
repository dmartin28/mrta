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

from phase2.IP_assignment import IP_assignment
from algorithms.cluster_assignment_nn import cluster_assignment_nn
import test_utils as tu

import torch
from ML.synergy_model import SynergyModel

"""HyperParameters"""
nu = 59 #number of robots # was 10
mu = 25 # number of tasks  # was 5
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task
L_t = 6 # Max number of tasks in a cluster # must be 6 to work with NN
L_r = 6 # Max number of robots in a cluster # must be 6 to work with NN
num_iterations = 100 # number of iterations to run
epsilon = 0.50 # probability of random merge of clusters

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
    'L_t': L_t,   # maximum number of tasks in a cluster
    'L_r': L_r,   # maximum number of robots in a cluster
    'epsilon': epsilon, # probability of random merge of clusters
}

# Define the environment size (Min is always 0)
max_x = 100
max_y = 100

# Load the saved model
# Will have size 264 when L_t = 6, L_r = 6, kappa = 2
model = SynergyModel(264)
model.load_state_dict(torch.load('best_linear_nn_model.pth'))
model.eval()

#Generate Problem Instance
robot_list, task_list = tu.generate_problem_instance(hypes, max_x, max_y)

# Perform cluster assignment using NN as guide
total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_nn(model, robot_list, task_list, num_iterations, hypes, printout=True)

# Print final results of all iterations
print("\n--- Final Results ---")
for i in range(len(iteration_rewards)):
    #print(f"Iteration {i + 1}:")
    #print(f"Total Reward: {iteration_rewards[i]}")
    #print(f"Assignment: {iteration_assignments[i]}")
    print(f"Iteration {i + 1}: Total Reward: {iteration_rewards[i]}")

# Calculate optimal assignment
if nu < 10 and mu < 6:
    optimal_assignment, optimal_reward = IP_assignment(robot_list, task_list, hypes)
    print(f"\nOptimal Reward: {optimal_reward}")
    print(f"Optimal Assignment: {optimal_assignment}")
print(f"Iterative Assignment: {iteration_assignments[-1]}")