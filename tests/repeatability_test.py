import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import pickle
from algorithms.cluster_assignment_rand import cluster_assignment_rand
import test_utils as tu

"""HyperParameters"""
nu = 30  # number of robots
mu = 20  # number of tasks
kappa = 2  # number of capabilities
L = 3  # maximum team size for a single task

# Define the environment size
max_x = 100
max_y = 100

"Test Parameters"
cluster_size = 6
num_runs = 10
num_iterations = 100  # number of iterations to run

# Define a dictionary of hyperparameters to send to functions
hypes = {
    'nu': nu,      # number of robots
    'mu': mu,      # number of tasks
    'kappa': kappa,   # number of capabilities
    'L': L,       # maximum team size
    'L_t': cluster_size,   # maximum number of tasks in a cluster
    'L_r': cluster_size,   # maximum number of robots in a cluster
}

# Initialize a list to store results for each run
results = []

# Generate Problem Instance
robot_list, task_list = tu.generate_problem_instance(hypes, max_x, max_y)

for run in range(num_runs):
    print(f"Run: {run+1}")

    # Perform random iterative assignment
    total_reward, iteration_assignments, iteration_rewards, iteration_times = cluster_assignment_rand(
        robot_list, task_list, num_iterations, hypes)
    
    # Store the iteration rewards for this run
    results.append(iteration_rewards)

# Save results to file
with open('random_cluster_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to 'random_cluster_results.pkl'")

# Plot the results
plt.figure(figsize=(12, 8))
for run, rewards in enumerate(results):
    # Create a new array with 0 at position 0, followed by the original rewards
    rewards_with_origin = np.insert(rewards, 0, 0)
    # Plot starting from iteration 0
    plt.plot(range(0, num_iterations + 1), rewards_with_origin, label=f'Run {run+1}')

plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title(f'Reward vs. Iteration for 5 Runs\n(Cluster Size: {cluster_size}, {nu} robots and {mu} tasks)')
plt.legend()
plt.grid(True)
plt.savefig('repeatability_test.png')
plt.show()