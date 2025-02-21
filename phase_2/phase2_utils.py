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

def calculate_net_reward(robot_team, task):
    if len(robot_team) < 1:
        return 0
    else:
        # Calculate capability value
        # print('Robot team', robot_team)
        # print('Robot team[0] capabilities', robot_team[0].get_capabilities())
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
        return net_reward

def partition_search(robots,tasks,partition):
    
    # Robot IDs assigned to the first task in tasks
    # Note this stores the global robot IDs, not indices
    t0_assignment = []

    # Base case: there is only one task
    if len(tasks) == 1:
        print('Final team_size_0: ', partition[0])
        print('len(robots): ', len(robots))
        print('len(tasks): ', len(tasks))
        for robot in robots:          
            t0_assignment.append([robot.get_id()])
        print('Final task Assignment: ', t0_assignment)    
        # Calculate net reward:
        reward = calculate_net_reward(robots, tasks[0])
        return t0_assignment,reward
    
    else:
        
        #Add case where size of team = 0, so there are no combos
        
        # Recursive case: there are multiple tasks
        team_size_0 = partition[0] # team size for the first task
        n = len(robots)
        max_reward = float('-inf')
        best_assignment = None
        
        # for all possible combinations of robots for the first task
        print('team_size_0: ', team_size_0)
        print('len(robots): ', len(robots))
        print('len(tasks): ', len(tasks))
        combos = combinations(range(n), team_size_0)
        for combo in combos:
            print("combo: ", combo)
            robots_t0 = [] # Stores robot indices in list
            t0_assignment = [] # Stores robot ids
            unused_robots = robots.copy()
            other_tasks = tasks.copy()
            other_tasks.pop(0)
            sub_partition = partition.copy()
            sub_partition.pop(0)
    
            # Create a list of indices to remove, sorted in descending order
            indices_to_remove = sorted(combo, reverse=True)
    
            # Remove robots from unused_robots and add them to robots_t0
            for robot_idx in indices_to_remove:
                robot = unused_robots.pop(robot_idx)
                robots_t0.append(robot)
                t0_assignment.append(robot.get_id())
    
            reward_0 = calculate_net_reward(robots_t0, tasks[0])
            sub_assignment, add_rewards = partition_search(unused_robots,other_tasks,sub_partition)
            reward = reward_0 + add_rewards
            
            if reward > max_reward:
                max_reward = reward
                best_assignment = [t0_assignment] + [sub_assignment]
                
            return best_assignment, max_reward
            
            
            
        

        
    # Generate all possible combinations of robots for the first team
