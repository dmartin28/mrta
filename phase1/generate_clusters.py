
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_classes.robot import Robot
from shared_classes.task import Task
import phase1.phase1_utils as utils
import random

def generate_clusters_merge(robot_list,task_list,L_r,L_t):
    
    kappa = len(robot_list[0].get_capabilities())

    #Initialize each robot + task in their individual cluster
    clusters = []
    for robot in robot_list:
        clusters.append([[robot.id], []])
    for task in task_list:
        clusters.append([[], [task.id]])
    # Each cluster is 2D array, row 1 are robot indices, row 2 are task ids

    # For each pair of clusters, calculate the value of the merged cluster
    equilibrium = False
    while equilibrium == False:
        max_change = 0
        merge_indices = []
        for i in range(len(clusters)):
            for j in range (i+1, len(clusters)):
            
                # print("Checking clusters: ", clusters[i], clusters[j])
                # print("Robot indices: ", clusters[i][0], clusters[j][0])
                # print("Task indices: ", clusters[i][1], clusters[j][1])
                # print("clusters[i][0]) + len(clusters[j][0])", len(clusters[i][0]) + len(clusters[j][0]))
                # print("clusters[i][1]) + len(clusters[j][1])", len(clusters[i][1]) + len(clusters[j][1]))
                # Check if the merge would exceed the maximum cluster sizes
                if len(clusters[i][0]) + len(clusters[j][0]) <= L_r and len(clusters[i][1]) + len(clusters[j][1]) <= L_t:
                    merged_cluster = [clusters[i][0] + clusters[j][0], clusters[i][1] + clusters[j][1]]
                    #print("Merged cluster: ", merged_cluster)
                    merged_value = utils.coalition_value([robot_list[r] for r in merged_cluster[0]], [task_list[t] for t in merged_cluster[1]], kappa)

                    #Can probably store these values so we don't recompute each time:
                    clusteri_val = utils.coalition_value([robot_list[r] for r in clusters[i][0]], [task_list[t] for t in clusters[i][1]], kappa)
                    clusterj_val = utils.coalition_value([robot_list[r] for r in clusters[j][0]], [task_list[t] for t in clusters[j][1]], kappa)
                    difference = merged_value - clusteri_val - clusterj_val

                    if difference > max_change:
                        max_change = difference
                        merge_indices = [i,j]

        if max_change > 0:
            # Merge the clusters
            clusters[merge_indices[0]][0] += clusters[merge_indices[1]][0]
            clusters[merge_indices[0]][1] += clusters[merge_indices[1]][1]
            clusters.pop(merge_indices[1])
        else:
            equilibrium = True
    return clusters

def refine_clusters_merge(initial_clusters, robot_list, task_list, L_r, L_t):
    # this function takes some exisiting clusters and refines them by merging them together.
    clusters = initial_clusters
    kappa = len(robot_list[0].get_capabilities())
    # For each pair of clusters, calculate the value of the merged cluster
    equilibrium = False
    while equilibrium == False:
        max_change = 0
        merge_indices = []
        for i in range(len(clusters)):
            for j in range (i+1, len(clusters)):
            
                # Check if the merge would exceed the maximum cluster sizes
                if len(clusters[i][0]) + len(clusters[j][0]) <= L_r and len(clusters[i][1]) + len(clusters[j][1]) <= L_t:
                    merged_cluster = [clusters[i][0] + clusters[j][0], clusters[i][1] + clusters[j][1]]
                    #print("Merged cluster: ", merged_cluster)
                    merged_value = utils.coalition_value([robot_list[r] for r in merged_cluster[0]], [task_list[t] for t in merged_cluster[1]], kappa)

                    # Can probably store these values so we don't recompute each time:
                    clusteri_val = utils.coalition_value([robot_list[r] for r in clusters[i][0]], [task_list[t] for t in clusters[i][1]], kappa)
                    clusterj_val = utils.coalition_value([robot_list[r] for r in clusters[j][0]], [task_list[t] for t in clusters[j][1]], kappa)
                    difference = merged_value - clusteri_val - clusterj_val

                    if difference > max_change:
                        max_change = difference
                        merge_indices = [i,j]

        if max_change > 0:
            # Merge the clusters
            clusters[merge_indices[0]][0] += clusters[merge_indices[1]][0]
            clusters[merge_indices[0]][1] += clusters[merge_indices[1]][1]
            clusters.pop(merge_indices[1])
        else:
            equilibrium = True
    return clusters

def refine_clusters_random_merge(initial_clusters, L_r, L_t):
    clusters = initial_clusters.copy()
    
    while True:
        # Create a list of all possible merge pairs
        merge_candidates = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                if (len(clusters[i][0]) + len(clusters[j][0]) <= L_r and 
                    len(clusters[i][1]) + len(clusters[j][1]) <= L_t):
                    merge_candidates.append((i, j))
        
        # If no more merges are possible, break the loop
        if not merge_candidates:
            break
        
        # Randomly select a merge pair
        merge_indices = random.choice(merge_candidates)
        
        # Merge the selected clusters
        i, j = merge_indices
        clusters[i][0] += clusters[j][0]
        clusters[i][1] += clusters[j][1]
        clusters.pop(j)
    
    return clusters

def nash_eq_clustering(robot_list, task_list, L_r):
    kappa = len(robot_list[0].get_capabilities())

    #Initialize each robot + task in their individual cluster
    clusters = []
    for robot in robot_list:
        clusters.append([[robot.id], []])
    for task in task_list:
        clusters.append([[], [task.id]])
    # Each cluster is 2D array, row 1 are robot indices, row 2 are task indices

    equilibrium = False
    while not equilibrium:
        equilibrium = True

        # Search through all potential robot moves
        best_move = None
        max_change = 0

        for i, source_cluster in enumerate(clusters):
            for j, robot_id in enumerate(source_cluster[0]):
                for k, target_cluster in enumerate(clusters):
                    if i != k and len(target_cluster[0]) < L_r:
                        # Calculate current values
                        current_value = utils.nash_eq_coalition_val([robot_list[r] for r in source_cluster[0]], [task_list[t] for t in source_cluster[1]], kappa,L_r)
                        target_value = utils.nash_eq_coalition_val([robot_list[r] for r in target_cluster[0]], [task_list[t] for t in target_cluster[1]], kappa,L_r)

                        # Calculate new values after potential move
                        new_source_value = utils.nash_eq_coalition_val([robot_list[r] for r in source_cluster[0] if r != robot_id], [task_list[t] for t in source_cluster[1]], kappa,L_r)
                        new_target_value = utils.nash_eq_coalition_val([robot_list[r] for r in target_cluster[0]] + [robot_list[robot_id]], [task_list[t] for t in target_cluster[1]], kappa,L_r)

                        # Calculate net change
                        change = (new_source_value + new_target_value) - (current_value + target_value)

                        if change > max_change:
                            max_change = change
                            best_move = (i, k, robot_id)

        # Perform the best robot move if it increases the net value
        if max_change > 0:
            equilibrium = False
            source_cluster_index, target_cluster_index, robot_id_to_move = best_move

            # Remove the robot from its current cluster
            clusters[source_cluster_index][0].remove(robot_id_to_move)

            # Add the robot to the target cluster
            clusters[target_cluster_index][0].append(robot_id_to_move)

            # If the source cluster is now empty, remove it
            if len(clusters[source_cluster_index][0]) == 0 and len(clusters[source_cluster_index][1]) == 0:
                clusters.pop(source_cluster_index)

    return clusters

def generate_clusters_move(robot_list, task_list, L_r, L_t):
    
    kappa = len(robot_list[0].get_capabilities())

    # Initialize each robot + task in their individual cluster
    clusters = []
    for robot in robot_list:
        clusters.append([[robot.id], []])
    for task in task_list:
        clusters.append([[], [task.id]])

    equilibrium = False
    while not equilibrium:
        equilibrium = True

        # Loop through all robots
        i = 0
        while i < len(clusters):
            cluster = clusters[i]
            j = 0
            while j < len(cluster[0]):
                robot_id = cluster[0][j]
                max_change = 0
                best_move = None

                # Check net gain caused by move to all other clusters
                for k, target_cluster in enumerate(clusters):
                    if i != k and len(target_cluster[0]) < L_r:
                        # Calculate current values
                        current_value = utils.coalition_value([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], kappa)
                        target_value = utils.coalition_value([robot_list[r] for r in target_cluster[0]], [task_list[t] for t in target_cluster[1]], kappa)

                        # Calculate new values after potential move
                        new_source_value = utils.coalition_value([robot_list[r] for r in cluster[0] if r != robot_id], [task_list[t] for t in cluster[1]], kappa)
                        new_target_value = utils.coalition_value([robot_list[r] for r in target_cluster[0]] + [robot_list[robot_id]], [task_list[t] for t in target_cluster[1]], kappa)

                        # Calculate net change
                        change = (new_source_value + new_target_value) - (current_value + target_value)

                        if change > max_change:
                            max_change = change
                            best_move = (k, robot_id)

                # Perform move that most increases the net cluster value
                if max_change > 0:
                    equilibrium = False
                    target_cluster_index = best_move[0]
                    robot_id_to_move = best_move[1]
    
                    # Remove the robot from its current cluster
                    cluster[0].remove(robot_id_to_move)
    
                    # Add the robot to the target cluster
                    clusters[target_cluster_index][0].append(robot_id_to_move)

                    # If the cluster is now empty, remove it
                    if len(cluster[0]) == 0 and len(cluster[1]) == 0:
                        clusters.pop(i)
                        i -= 1
                        break
                else:
                    j += 1
            i += 1

        # Loop through all tasks
        i = 0
        while i < len(clusters):
            cluster = clusters[i]
            j = 0
            while j < len(cluster[1]):
                task_id = cluster[1][j]
                max_change = 0
                best_move = None

                # Check net gain caused by move to all other clusters
                for k, target_cluster in enumerate(clusters):
                    if i != k and len(target_cluster[1]) < L_t:
                        # Calculate current values
                        current_value = utils.coalition_value([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1]], kappa)
                        target_value = utils.coalition_value([robot_list[r] for r in target_cluster[0]], [task_list[t] for t in target_cluster[1]], kappa)

                        # Calculate new values after potential move
                        new_source_value = utils.coalition_value([robot_list[r] for r in cluster[0]], [task_list[t] for t in cluster[1] if t != task_id], kappa)
                        new_target_value = utils.coalition_value([robot_list[r] for r in target_cluster[0]], [task_list[t] for t in target_cluster[1]] + [task_list[task_id]], kappa)

                        # Calculate net change
                        change = (new_source_value + new_target_value) - (current_value + target_value)

                        if change > max_change:
                            max_change = change
                            best_move = (k, task_id)

                # Perform move that most increases the net cluster value
                if max_change > 0:
                    equilibrium = False
                    target_cluster_index = best_move[0]
                    task_id_to_move = best_move[1]
    
                    # Remove the task from its current cluster
                    cluster[1].remove(task_id_to_move)
    
                    # Add the task to the target cluster
                    clusters[target_cluster_index][1].append(task_id_to_move)

                    # If the cluster is now empty, remove it
                    if len(cluster[0]) == 0 and len(cluster[1]) == 0:
                        clusters.pop(i)
                        i -= 1
                        break
                else:
                    j += 1
            i += 1

    return clusters
