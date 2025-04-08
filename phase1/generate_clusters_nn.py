
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_classes.robot import Robot
from shared_classes.task import Task
import phase1.phase1_utils as utils
import random
import torch
import numpy as np

def refine_clusters_random_merge(initial_clusters, L_r, L_t, model, epsilon=0.1):
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
                    
        ############# Need to build this function to create the input vector
                    input_vector = create_input_vector(clusters[cluster_index], clusters[candidate])
                    
        ############# Need to figure out how to load the model
                    # Predicted reward = model(input_vector)
                    with torch.no_grad():
                        predicted_reward = model(torch.FloatTensor(input_vector)).item()
                    predicted_rewards.append(predicted_reward)
                
                # Create a softmax distribution over the predicted rewards
                softmax_probs = np.exp(predicted_rewards) / np.sum(np.exp(predicted_rewards))
                
                # Choose a merge pair based on the softmax distribution
                merge_index = np.random.choice(merge_candidates, p=softmax_probs)
            
            # Perform the merge
            clusters[cluster_index][0] += clusters[merge_index][0]
            clusters[cluster_index][1] += clusters[merge_index][1]
            clusters.pop(merge_index)
            
            # Update incomplete_clusters and complete_clusters
            # I don't think this test is necessary.
            if len(clusters[cluster_index][0]) == L_r and len(clusters[cluster_index][1]) == L_t:
                complete_clusters.append(cluster_index)
                incomplete_clusters.remove(cluster_index)
            # This is necessary, but should always be true right???
            if merge_index in incomplete_clusters:
                incomplete_clusters.remove(merge_index)
            
            # Adjust indices in incomplete_clusters and complete_clusters
            incomplete_clusters = [i if i < merge_index else i-1 for i in incomplete_clusters]
            complete_clusters = [i if i < merge_index else i-1 for i in complete_clusters]
    
    return clusters

def create_input_vector(cluster1, cluster2):
    # Implement this function to create the input vector for the neural network
    # based on the two clusters that are candidates for merging
    pass


# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from shared_classes.robot import Robot
# from shared_classes.task import Task
# import phase1.phase1_utils as utils
# import random

# def refine_clusters_random_merge(initial_clusters, L_r, L_t):
#     clusters = initial_clusters.copy()
    
#     # create list of clusters that can still increase in size without exceeding L_r or L_t
#     # incomplete_clusters = []
    
#     # create list of clsters that are already at max size
#     # complete_clusters = []

#     # while there are some incomplete cluster
#         # randomly choose one of the incomplete clusters
#         # find all possible merge pairs for this cluster that do not exceed L_r or L_t

#         # if there are no possible merge pairs, move this cluster to the complete_clusters list
#         # Else:
#             # epsilon  percent of the time choose a random merge pair
#             # 1-epsilon percent of the time, choose merge pair with the highest predicted reward
#                 # create NN input vector for each pair of possible merges
#                 # predicted reward = model(input_vector)
#                 # create a softmax distribution over the predicted rewards
#                 # choose a merge pair based on the softmax distribution
#         # 

#     # Old code commented out
#     # while True:
#     #     # Create a list of all possible merge pairs
#     #     merge_candidates = []
#     #     for i in range(len(clusters)):
#     #         for j in range(i+1, len(clusters)):
#     #             if (len(clusters[i][0]) + len(clusters[j][0]) <= L_r and 
#     #                 len(clusters[i][1]) + len(clusters[j][1]) <= L_t):
#     #                 merge_candidates.append((i, j))
        
#     #     # If no more merges are possible, break the loop
#     #     if not merge_candidates:
#     #         break
        
#     #     # Randomly select a merge pair
#     #     merge_indices = random.choice(merge_candidates)
        
#     #     # Merge the selected clusters
#     #     i, j = merge_indices
#     #     clusters[i][0] += clusters[j][0]
#     #     clusters[i][1] += clusters[j][1]
#     #     clusters.pop(j)
    
#     return clusters