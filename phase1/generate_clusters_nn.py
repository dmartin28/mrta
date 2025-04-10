
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_classes.robot import Robot
from shared_classes.task import Task
import phase1.phase1_utils as utils
import random
import torch
import numpy as np

def refine_clusters_nn_merge(initial_clusters,robot_list, task_list, L_r, L_t, kappa, L, model, epsilon):
    
    clusters = initial_clusters.copy()
    
    # Create list of clusters that can still increase in size without exceeding L_r or L_t
    incomplete_clusters = [i for i, cluster in enumerate(clusters) 
                           if len(cluster[0]) < L_r or len(cluster[1]) < L_t]
    
    # Create list of clusters that are already at max size
    complete_clusters = [i for i, cluster in enumerate(clusters) 
                         if len(cluster[0]) == L_r and len(cluster[1]) == L_t]

    while incomplete_clusters:
        # Randomly choose one of the incomplete clusters
        cluster_index = random.choice(incomplete_clusters)
        
        # Find all possible merge pairs for this cluster that do not exceed L_r or L_t
        merge_candidates = []
        for j in incomplete_clusters:
            if j != cluster_index:
                if (len(clusters[cluster_index][0]) + len(clusters[j][0]) <= L_r and 
                    len(clusters[cluster_index][1]) + len(clusters[j][1]) <= L_t):
                    merge_candidates.append(j)

        if not merge_candidates:
            # If there are no possible merge pairs, move this cluster to the complete_clusters list
            complete_clusters.append(cluster_index)
            incomplete_clusters.remove(cluster_index)
        else:
            if random.random() < epsilon:
                # With probability epsilon choose a random merge pair
                merge_index = random.choice(merge_candidates)
            else:
                # With prob 1-epsilon choose merge pair with the highest predicted reward
                predicted_rewards = []
                for candidate in merge_candidates:
                    # Create NN input vector for each pair of possible merges
                    
                    # Need to build this function to create the input vector
                    input_vector = create_input_vector(clusters[cluster_index], clusters[candidate],robot_list,task_list, kappa,L,L_r,L_t)
                    # print(f"input_vector shape: {input_vector.shape}")
                    # Need to figure out how to load the model
                    # Predicted reward = model(input_vector)
                    with torch.no_grad():
                        #predicted_reward = model(torch.FloatTensor(input_vector)).item()
                        predicted_reward = model(torch.FloatTensor(input_vector)).item()
                    predicted_rewards.append(predicted_reward)
                # Create a softmax distribution over the predicted rewards
                softmax_probs = stable_softmax(predicted_rewards)
                print(f"Predicted rewards: {[round(reward, 2) for reward in predicted_rewards]}")
                print(f"Softmax probabilities: {[round(prob, 2) for prob in softmax_probs]}")
                
                # Choose a merge pair based on the softmax distribution
                merge_index = np.random.choice(merge_candidates, p=softmax_probs)
            
            # Perform the merge
            clusters[cluster_index][0] += clusters[merge_index][0]
            clusters[cluster_index][1] += clusters[merge_index][1]
            clusters.pop(merge_index)
            
            # Update incomplete_clusters and complete_clusters
            # I don't think this test is necessary.
            # if len(clusters[cluster_index][0]) == L_r and len(clusters[cluster_index][1]) == L_t:
            #     complete_clusters.append(cluster_index)
            #     incomplete_clusters.remove(cluster_index)
            
            # This is necessary, but should always be true right???
            if merge_index in incomplete_clusters:
                incomplete_clusters.remove(merge_index)
            
            # Adjust indices in incomplete_clusters and complete_clusters
            incomplete_clusters = [i if i < merge_index else i-1 for i in incomplete_clusters]
            complete_clusters = [i if i < merge_index else i-1 for i in complete_clusters]
    
    return clusters

def create_input_vector(cluster1, cluster2, robot_list, task_list, kappa,L,L_r,L_t):
    
    robots_1 = cluster1[0]
    tasks_1 = cluster1[1]
    robots_2 = cluster2[0]
    tasks_2 = cluster2[1]
    
    # Define the size of robot and task information
    robot_info_size = 2 + kappa  # x, y, and kappa capability values
    task_info_size = 2 + (L+1)**kappa  # x, y, and flattened reward matrix

    # Concatenate all of cluster 1 info into a vector
    cluster_1_vector = []
    for robot_idx in robots_1:
        robot = robot_list[robot_idx]
        cluster_1_vector.extend([robot.x, robot.y] + robot.capabilities.tolist())
    # Pad with zeros if there are fewer than L_r robots
    cluster_1_vector.extend([0] * (L_r - len(robots_1)) * robot_info_size)

    for task_idx in tasks_1:
        task = task_list[task_idx]
        cluster_1_vector.extend([task.x, task.y] + task.reward_matrix.flatten().tolist())
    # Pad with zeros if there are fewer than L_t tasks
    cluster_1_vector.extend([0] * (L_t - len(tasks_1)) * task_info_size)

    # Concatenate all of cluster 2 info into a vector
    cluster_2_vector = []
    for robot_idx in robots_2:
        robot = robot_list[robot_idx]
        cluster_2_vector.extend([robot.x, robot.y] + robot.capabilities.tolist())
    # Pad with zeros if there are fewer than L_r robots
    cluster_2_vector.extend([0] * (L_r - len(robots_2)) * robot_info_size)

    for task_idx in tasks_2:
        task = task_list[task_idx]
        cluster_2_vector.extend([task.x, task.y] + task.reward_matrix.flatten().tolist())
    # Pad with zeros if there are fewer than L_t tasks
    cluster_2_vector.extend([0] * (L_t - len(tasks_2)) * task_info_size)

    # Concatenate everything into one vector (This is the input to the NN)
    # Will have size 264 when L_t = 6, L_r = 6, kappa = 2
    nn_input = np.array(cluster_1_vector + cluster_2_vector, dtype=np.float32)

    print(f"nn_input: {nn_input}")
    return nn_input

def stable_softmax(x):
    x = np.array(x)
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x)