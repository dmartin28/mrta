import numpy as np
import math
from itertools import combinations
from robot import Robot

# My Code:
import phase2_utils as utils

def IP_assignment_robust(robot_list, task_list, L,kappa):
    
    n = len(robot_list)
    m = len(task_list)

    # Check if there are no tasks or robots
    if n == 0:
        best_assignment = [ [] for _ in range(m)]
        reward = 0
        return best_assignment, reward
    if m == 0:
        best_assignment = [[robot.get_id() for robot in robot_list]]
        reward = 0
        return best_assignment, reward
    

    best_assignment = None
    global_reward = float('-inf')

    # Generate all possible numbers of unused robots (0 to n)
    # For each, build the partitions of the rest of tasks

    partitions = utils.generate_partitions_robust(n,m,L)



    print(f"All partitions of {n} into {m}+1 parts with first part unassigned:")
    for partition in partitions:
        print(partition)
    print(f"Total number of partitions: {len(partitions)}")


    # Calculate mean and average reward for each each for each possible team size:
    max_team_size = L
    max_rewards = np.full((max_team_size+1, m), float('-inf'))
    avg_rewards = np.zeros((max_team_size+1, m))

    # Calculate max and average reward for each team size for each task
    for team_size in range(0, max_team_size+1):
        for task_idx in range(0, m):
            # Generate combinations
            combos = combinations(range(n), team_size)
            num_combos = math.comb(n, team_size)

            for combo in combos:                        
                #Calculate capability value
                team_capabilities = np.zeros(kappa, dtype=np.int32)
                for robot_idx in combo:
                    team_capabilities += robot_list[robot_idx].get_capabilities()
                capability_value =  task_list[task_idx].get_reward(*team_capabilities)

                # Calculate cost assuming cost = distance traveled
                cost = 0
                for robot_idx in combo:
                    cost += math.dist(robot_list[robot_idx].get_location(), task_list[task_idx].get_location())
                
                # Here is where you can change the reward function
                net_reward = capability_value - cost
                # net_reward = capability_value
                
                if net_reward > max_rewards[team_size, task_idx]:
                    max_rewards[team_size, task_idx] = net_reward
                
                avg_rewards[team_size, task_idx] += net_reward/num_combos

    # Print the average rewards matrix
    print("Average Rewards Matrix: Rows = Team Size, Columns = Task")
    print(avg_rewards)
    print()

    # Print the maximum rewards matrix
    print("Maximum Rewards Matrix: Rows = Team Size, Columns = Task")
    print(max_rewards)

    # Calculate upper and lower bounds for each partition
    upper_bounds = np.zeros(len(partitions))
    lower_bounds = np.zeros(len(partitions))
    for partition_idx in range(0,len(partitions)):
        for task_idx in range(0, m):
            # Have to add 1 to task_idx because partitions start with unassigned robots T0
            lower_bounds[partition_idx] += avg_rewards[partitions[partition_idx][task_idx+1], task_idx]
            upper_bounds[partition_idx] += max_rewards[partitions[partition_idx][task_idx+1], task_idx]

    # Write partitions with their upper and lower bounds
    print("\nPartitions with Upper and Lower Bounds:")
    print("Partition | Lower Bound | Upper Bound")
    print("-" * 40)
    for partition_idx, partition in enumerate(partitions):
        print(f"{partition} | {lower_bounds[partition_idx]:.2f} | {upper_bounds[partition_idx]:.2f}")
    #######################################################
    print(f"Total number of partitions: {len(partitions)}")


    """
    Here we are testing the calculate =_upper_bounds_function
    """

    assigned_partitions = [partition[1:] for partition in partitions]
    print("\nAssigned Partitions: ", assigned_partitions)
    upper_bounds_test = utils.calculate_upper_bounds(robot_list, task_list, assigned_partitions,L)
    
    # Write partitions with their upper and lower bounds
    print("\nPartitions with Upper Bounds:")
    print("Partition | Upper Bound")
    print("-" * 40)
    for partition_idx, partition in enumerate(assigned_partitions):
        print(f"{partition} |  {upper_bounds_test[partition_idx]:.2f}")
    #######################################################
    print(f"Total number of partitions: {len(partitions)}")





    # Find global upper and lower bounds
    LB = max(lower_bounds)
    # UB = max(upper_bounds)

    while len(partitions) > 0:
        
        best_partition_idx = np.argmax(upper_bounds)
        
        # Print best partition
        print("\nBest Partition:")
        print("Partition | Lower Bound | Upper Bound")
        print("-" * 40)
        print(f"{partitions[best_partition_idx]} | {lower_bounds[best_partition_idx]:.2f} | {upper_bounds[best_partition_idx]:.2f}")
        
        # Search partition with highest UB
        #partition_without_dummy_task = partitions[best_partition_idx][1:]
        partition_assignment, partition_reward = utils.partition_search_dummyTask(robot_list, task_list, partitions[best_partition_idx])
        
        # Remove best partition from list
        partitions.pop(best_partition_idx)
        upper_bounds = np.delete(upper_bounds, best_partition_idx)
        
        if partition_reward > global_reward:
            global_reward = partition_reward
            best_assignment = partition_assignment
            print("LB before: ", LB)
            LB = max(LB,global_reward)
            print("LB after: ", LB)
            
        print("\nbest_assignment: ", best_assignment)
        print("global_reward: ", global_reward)
        print("LB: ", LB)
        
        # Trim all partitions whose UB is less than the LB
        for partition_idx in range(len(partitions) - 1, -1, -1):
            if upper_bounds[partition_idx] < LB:
                del partitions[partition_idx]
                upper_bounds = np.delete(upper_bounds, partition_idx)
                lower_bounds = np.delete(lower_bounds, partition_idx)
                
        ##### Write trimmed partitions with their upper and lower bounds ####
        print("\nTrimmed Partitions with Upper and Lower Bounds:")
        print("Partition | Lower Bound | Upper Bound")
        print("-" * 40)
        for partition_idx, partition in enumerate(partitions):
            print(f"{partition} | {lower_bounds[partition_idx]:.2f} | {upper_bounds[partition_idx]:.2f}")
        ######################################################################   
        
        print("num partitions left: ", len(partitions))
        print("num_upper_bounds left: ", len(upper_bounds))
        
    print("Final best_assignment: ", best_assignment)
    print("Final global_reward: ", round(global_reward))
    
    return best_assignment, global_reward