
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_classes.robot import Robot
from shared_classes.task import Task
import phase1.phase1_utils as utils

def generate_clusters(robot_list,task_list,L_r,L_t):
    
    kappa = len(robot_list[0].get_capabilities())

    #Initialize each robot + task in their individual cluster
    clusters = []
    for robot in robot_list:
        clusters.append([[robot.id], []])
    for task in task_list:
        clusters.append([[], [task.id]])
    # Each cluster is 2D array, row 1 are robot indices, row 2 are task indices

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