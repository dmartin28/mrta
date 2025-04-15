
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

""" Refine clusters by merging them randomly until no more merges are possible."""
def refine_clusters_random_merge(initial_clusters, L_r, L_t):
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
            # Choose a random merge pair
            merge_index = random.choice(merge_candidates)
            
            # Perform the merge
            clusters[cluster_index][0] += clusters[merge_index][0]
            clusters[cluster_index][1] += clusters[merge_index][1]
            clusters.pop(merge_index)
            
            # Remove the merged cluster from incomplete_clusters
            if merge_index in incomplete_clusters:
                incomplete_clusters.remove(merge_index)
            
            # Adjust indices in incomplete_clusters and complete_clusters
            incomplete_clusters = [i if i < merge_index else i-1 for i in incomplete_clusters]
            complete_clusters = [i if i < merge_index else i-1 for i in complete_clusters]
    
    return clusters