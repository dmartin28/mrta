from itertools import combinations
import numpy as np
import math

def generate_partitions(n, m, L):
    def partition_helper(n, m, L, current_partition, partitions):
        if m == 0:
            if n == 0:
                partitions.append(current_partition[:])
            return

        for i in range(0, min(n, L) + 1):
            current_partition.append(i)
            partition_helper(n - i, m - 1, L, current_partition, partitions)
            current_partition.pop()

    partitions = []
    partition_helper(n, m, L, [], partitions)
    return partitions

def calculate_net_reward(robot_team, task)
    # Calculate capability value
    team_capabilities = np.zeros(len(robot_team[0].get_capabilities()), dtype=np.int32)
    for robot_idx in range(len(robot_team)):
        team_capabilities += robot_team[robot_idx].get_capabilities()
    capability_value =  task.get_reward(*team_capabilities)
    
    # Calculate cost assuming cost = distance traveled                
    cost = 0
    for robot_idx in range(len(robot_team)):
        cost += math.dist(robot_team[robot_idx].get_location(), task.get_location())

    # net_reward = capability_value - cost
    net_reward = capability_value

def partition_search(robots,tasks,partition)
    
    # Robot IDs assigned to the first task in tasks
    # Note this stores the global robot IDs, not indices
    t0_assignment = []

    # Base case: there is only one task
    if len(tasks) == 1:
        for robot in robots:          
            t0_assignment.append([robot.get_id()])
            
        # Calculate net reward:
        reward = calculate_net_reward(robots, tasks[0])
        return t0_assignment
    
    else:
        # Recursive case: there are multiple tasks
        team_size_0 = partition[0] # team size for the first task
        

        # for all possible combinations of robots for the first task

        
    # Generate all possible combinations of robots for the first team
