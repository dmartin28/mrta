"""
This script generates a dataset where each entry includes information about 
a pair of clusters, along with a target value representing the difference in rewards between 
the combined optimal assignment and the individual optimal assignments of the clusters.
This dataset will be used to train a neural network to predict the reward of merging two clusters.
"""

import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import random
from shared_classes.task import Task
from shared_classes.robot import Robot
from phase2.IP_assignment import IP_assignment
import ML.ML_utils as utils
import tests.test_utils as test_utils
from sklearn.model_selection import train_test_split

""" Desired dataset size """
number_of_samples = 100000  # number of cluster pairs to produce

"""HyperParameters"""
kappa = 2 # number of different capabilities
L = 3  # maximum team size for a single task
L_t = 6 # Max number of tasks in a cluster
L_r = 6 # Max number of robots in a cluster

""" Define the environment size"""
max_x = 100
min_x = 0
max_y = 100
min_y = 0

""" Define some specific task types: """
task_types = test_utils.generate_task_types(L, kappa)

""" Define the two robot types: """
robot_type_1 = [1,0] # Note this is the capability vector
robot_type_2 = [0,1]

# Initialize list to store all data that will be outputted to a dataset
# Not the most efficient way to do this, but it works for now
all_data = []

for sample in range(number_of_samples):
    
    # Generate random number of tasks and robots for each cluster
    (num_tasks_1, num_robots_1), (num_tasks_2, num_robots_2) = utils.generate_cluster_sizes(L_r, L_t)

    # Generate tasks and robots for cluster 1
    tasks_1 = []
    for i in range(num_tasks_1):
        task_type = random.choice(list(task_types.values()))
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        tasks_1.append(Task(i, task_type, x, y))

    robots_1 = []
    for i in range(num_robots_1):
        robot_type = random.choice([robot_type_1, robot_type_2])
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        robots_1.append(Robot(i, robot_type, x, y))

    # Generate tasks and robots for cluster 2
    tasks_2 = []
    for i in range(num_tasks_2):
        task_type = random.choice(list(task_types.values()))
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        tasks_2.append(Task(i + num_tasks_1, task_type, x, y))

    robots_2 = []
    for i in range(num_robots_2):
        robot_type = random.choice([robot_type_1, robot_type_2])
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        robots_2.append(Robot(i + num_robots_1, robot_type, x, y))

    # Define the size of vector needed to store robot and task information
    robot_info_size = 2 + kappa  # x, y, and kappa capability values
    task_info_size = 2 + (L+1)**kappa  # x, y, and flattened reward matrix

    """ Concatenate all of cluster 1 info into a vector """
    cluster_1_vector = []
    for robot in robots_1:
        cluster_1_vector.extend([robot.x, robot.y] + robot.capabilities.tolist())
    # Pad with zeros if there are fewer than L_r robots
    cluster_1_vector.extend([0] * (L_r - len(robots_1)) * robot_info_size)

    for task in tasks_1:
        cluster_1_vector.extend([task.x, task.y] + task.reward_matrix.flatten().tolist())
    # Pad with zeros if there are fewer than L_t tasks
    cluster_1_vector.extend([0] * (L_t - len(tasks_1)) * task_info_size)

    """ Concatenate all of cluster 2 info into a vector """
    cluster_2_vector = []
    for robot in robots_2:
        cluster_2_vector.extend([robot.x, robot.y] + robot.capabilities.tolist())
    # Pad with zeros if there are fewer than L_r robots
    cluster_2_vector.extend([0] * (L_r - len(robots_2)) * robot_info_size)

    for task in tasks_2:
        cluster_2_vector.extend([task.x, task.y] + task.reward_matrix.flatten().tolist())
    # Pad with zeros if there are fewer than L_t tasks
    cluster_2_vector.extend([0] * (L_t - len(tasks_2)) * task_info_size)

    # Concatenate everything into one vector (This is will be the input to the NN)
    # Will have size 264 when L_t = 6, L_r = 6, kappa = 2
    nn_input = np.array(cluster_1_vector + cluster_2_vector, dtype=np.float32)

    #Calculate rewards using optimal assignment
    _, reward_1 = IP_assignment(robots_1, tasks_1, L, kappa)
    _, reward_2 = IP_assignment(robots_2, tasks_2, L, kappa)
    _, reward_combined = IP_assignment(robots_1 + robots_2, tasks_1 + tasks_2, L, kappa)

    # Target value for this sample
    target = reward_combined - (reward_1 + reward_2)

    # Append this sample to all_data
    all_data.append(np.concatenate((nn_input, [target])))

    # Print debugging information (you might want to remove or comment this out for large datasets)
    print(f"Sample {sample + 1}/{number_of_samples} generated")
    # print(f"Sample {sample + 1} nn_input shape: {nn_input.shape}")
    # print(f"Sample {sample + 1} target: {target}, type: {type(target)}")

    # # Print the input and target for debugging
    # # print(f"\nInput: {nn_input}")
    # # print("--------------------------------------------------")
    # # print(f"\nTarget: {target}")
    # print("--------------------------------------------------")
    # print(f"Cluster 1 Reward: {reward_1}")
    # print(f"Cluster 1 num_tasks: {num_tasks_1}")
    # print(f"Cluster 1 num_robots: {num_robots_1}")
    # print(f"Cluster 2 Reward: {reward_2}")
    # print(f"Cluster 2 num_tasks: {num_tasks_2}")
    # print(f"Cluster 2 num_robots: {num_robots_2}")
    # print(f"Combined Reward: {reward_combined}")
    # print("==================================================")

# Convert all_data to a numpy array
all_data = np.array(all_data)

# Create a DataFrame
columns = [f'feature_{i}' for i in range(264)] + ['target']
df = pd.DataFrame(all_data, columns=columns)

# Split the data into train and validation sets (80-20 split)
train_data, val_data = train_test_split(df, test_size=0.2, random_state=42)

# Create directories if they don't exist
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

# Save the datasets
train_data.to_csv('data/train/mrta_train.csv', index=False)
val_data.to_csv('data/val/mrta_val.csv', index=False)

# Save metadata
metadata = {
    'total_samples': len(df),
    'train_samples': len(train_data),
    'val_samples': len(val_data),
    'input_features': 264,
    'output_features': 1,
    'feature_description': 'First 264 columns represent robot and task information, last column is the target reward difference',
    'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'kappa': kappa,
    'L': L,
    'L_t': L_t,
    'L_r': L_r,
    'environment_size': f"{min_x}-{max_x}, {min_y}-{max_y}"
}

with open('data/mrta_dataset_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Dataset generation complete. Files saved in 'data' directory.")