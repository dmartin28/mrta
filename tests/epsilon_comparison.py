"""
This file compares the results of the random merging method with different group size limits
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import torch
from shared_classes.task import Task
from shared_classes.robot import Robot
from ML.synergy_model import SynergyModel
import matplotlib.pyplot as plt
import pickle
from algorithms.cluster_assignment_nn import cluster_assignment_nn
import test_utils as tu

"""HyperParameters"""
nu = 20 #number of robots
mu = 11 # number of tasks 
kappa = 2 # number of capabilities
L = 3 # maximum team size for a single task

# Define the environment size
max_x = 100
max_y = 100

"Test Parameters"
cluster_size = 6
epsilons = [0, 0.1, 0.5, 0.9, 1.0]
num_tests = 100
num_iterations = 50 # number of iterations to run

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
    'L_t': cluster_size,   # maximum number of tasks in a cluster
    'L_r': cluster_size,   # maximum number of robots in a cluster
}

# Initialize a dictionary to store results for each cluster size
results = {size: [] for size in epsilons}

# Load the saved model
# Will have size 264 when L_t = 6, L_r = 6, kappa = 2
model = SynergyModel(264)
model.load_state_dict(torch.load('best_linear_nn_model.pth'))
model.eval()

for test in range(num_tests):
    print(f"Test: {test+1}")

    """ Create a random problem instance """
    robot_list, task_list = tu.generate_problem_instance(hypes, max_x, max_y) 

    for epsilon in epsilons:
        print(f"Testing epsilon: {epsilon}")
        hypes['epsilon'] = epsilon  # Update the epsilon value in hyperparameters
        total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_nn(model, robot_list, task_list, num_iterations, hypes)
        results[epsilon].append(iteration_rewards)

# Calculate average iteration rewards for each cluster size across all tests
avg_results = {size: np.mean(np.array(rewards), axis=0) for size, rewards in results.items()}

# Save results and avg_results to files
with open('NN_epsilon_results.pkl', 'wb') as f:
    pickle.dump(results, f)

with open('NN_epsilon_avg_results.pkl', 'wb') as f:
    pickle.dump(avg_results, f)

print("Results saved to 'NN_epsilon_results.pkl' and 'NN_epsilon_results_plot.png'")

# Plot the results
plt.figure(figsize=(12, 8))
for size, avg_rewards in avg_results.items():
    # Create a new array with 0 at position 0, followed by the original rewards
    rewards_with_origin = np.insert(avg_rewards, 0, 0)
    # Plot starting from iteration 0
    plt.plot(range(0, num_iterations + 1), rewards_with_origin, label=f'Epsilon: {size}')

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title(f'Average Reward vs. Iteration for Different Epsilons \n(Averaged over {num_tests} tests with {nu} robots and {mu} tasks)')
plt.legend()
plt.grid(True)
plt.savefig('NN_epsilon_results_plot.png')
plt.show()