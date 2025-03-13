import numpy as np
import itertools
import math

from shared_classes.task import Task
from shared_classes.robot import Robot
import phase2.phase2_utils as phase2_utils

def get_coalition_capabilities(robots,kappa):
    # Sum up robot capabilities
    coalition_capabilities = np.zeros(kappa, dtype=int)
    for robot in robots:
        coalition_capabilities += robot.get_capabilities().astype(int)
    return coalition_capabilities

def find_max_index_within_limits(arr, limits):
    # Create a mask for valid indices
    mask = np.ones_like(arr, dtype=bool)
    for i, limit in enumerate(limits):
        index_arr = np.indices(arr.shape)[i]
        mask &= (index_arr <= limit)
    
    # Apply the mask to the array
    masked_arr = np.ma.array(arr, mask=~mask)
    
    # Find the index of the maximum value
    max_index = np.unravel_index(np.ma.argmax(masked_arr), arr.shape)
    
    return max_index

# Shapley based capability matching
def shapley_capability_matching(robots,tasks,kappa):

    # Create shapley vectors
    shapley_vectors = [np.array([]) for _ in range(kappa)]
    for task in tasks:
        task_shapleys = task.get_grand_shapley_vectors()
        for i in range(kappa):
            shapley_vectors[i] = np.concatenate([shapley_vectors[i],task_shapleys[i]])
    
    # Sort shapley vectors in descending order
    for i in range(kappa):
        shapley_vectors[i] = np.sort(shapley_vectors[i])[::-1]
    
    # Sum up robot capabilities
    coalition_capabilities = np.zeros(kappa, dtype=int)
    for robot in robots:
        coalition_capabilities += robot.get_capabilities().astype(int)
    
    # Calculate the total reward from capability matching
    capability_reward = 0
    for capability in range(kappa):
        # For each instance of the capability, add the shapley value
        for j in range(min(coalition_capabilities[capability], len(shapley_vectors[capability]))):
            capability_reward += shapley_vectors[capability][j]

    return capability_reward

# This just guarantees that the task can be completed with the given capabilities
def lower_bound_cap_matching(robots,tasks,kappa,print_out=False):
    
    # Sum up robot capabilities
    coalition_capabilities = np.zeros(kappa, dtype=int)
    for robot in robots:
        coalition_capabilities += robot.get_capabilities().astype(int)
    
    if print_out:
        print("\n New Call: -------------------------------")
        print("coalition_capabilities: ", coalition_capabilities)

    loop = True
    incomplete_tasks = tasks.copy()
    cap_reward = 0
    while loop:
        if print_out:
            print("\nStart Loop:")
            print("Incomplete Tasks: ", len(incomplete_tasks))
        # Search the tasks for the best that you can complete
        loop = False #will be set to True if we need to loop again
        max_reward = 0
        max_team = np.zeros(kappa)
        max_task_idx = -1 #initialize to an invalid index for error catching

        # Search the tasks for the best that you can complete
        for task_idx in range(len(incomplete_tasks)):      
            # Determine best capabilities for task
            reward_matrix = incomplete_tasks[task_idx].get_reward_matrix()
            best_team = find_max_index_within_limits(reward_matrix, coalition_capabilities)
            task_reward = reward_matrix[best_team]
            if print_out:
                print("Task: ", task_idx, " Reward Matrix: ")
                print(reward_matrix)
                print("best_capabilities: ", best_team)
            if task_reward > max_reward:
                max_reward = task_reward
                max_team = best_team
                max_task_idx = task_idx
        
        if max_reward > 0:
            # Add reward and remove task index from list of uncompleted tasks, subtract capabilities
            cap_reward += max_reward
            coalition_capabilities -= np.array(max_team)
            incomplete_tasks.pop(task_idx)
            if print_out:
                print("max_reward: ", max_reward, )
                print(f"Task {task_idx} assigned to team {best_team}")
                print("updated coalition_capabilities: ", coalition_capabilities)
            loop = True
    if print_out:
        print("Cap Reward: ", cap_reward,)
        print("End Call: -------------------------------\n")
    return cap_reward

def avg_distance_cost(robots,tasks):
    est_cost = 0
    m = len(tasks)
    for robot in robots:
        for task in tasks:
            est_cost += (1/m) * np.linalg.norm(np.array(robot.get_location()) - np.array(task.get_location()))
    return est_cost

def flexibility_reward(robots, tasks, kappa):
    
    # Can be thought of as average number of tasks each robot can be assigned to
    
    if len(robots)==0 or len(tasks)==0:
        return 0
    else:
        # Calculate the total number of tasks that use each capability
        task_count = np.zeros(kappa)
        for task in tasks:
            task_count += (task.get_grand_coalition() > 0).astype(int)
    
        # For each robot, add flexibility reward = sum of tasks robot could work on
        flex_reward = 0
        for robot in robots:
            robot_capabilities = robot.get_capabilities()
            robot_task_count = np.sum(task_count[robot_capabilities > 0])
            flex_reward += robot_task_count

        return flex_reward

def task_overlap(tasks,kappa):
    task_overlap = 0
    m = len(tasks)
    for i in range(m):
        for j in range(i+1,m):
            grand_coalition_i = tasks[i].get_grand_coalition()
            grand_coalition_j = tasks[j].get_grand_coalition()
            for k in range(kappa):
                task_overlap += min(grand_coalition_i[k], grand_coalition_j[k])


# Coalition evaluation function for nash equilibrium approach:
def nash_eq_coalition_val(robots, tasks, kappa,L):
    
    # Should not have more than one task:
    if len(tasks)==0:
        return 0
    
    if len(tasks) > 1:
        raise ValueError("Error, should not have more than one task")
    
    task = tasks[0]
    best_net_reward = 0
    best_team = []
    
    # Generate all possible combinations of robots
    for team_size in range(1, min(len(robots) + 1,L+1)): # need to add +1???
        for robot_combo in itertools.combinations(range(len(robots)), team_size):
            
            robot_team = [robots[robot_idx] for robot_idx in robot_combo]
            
            # Calculate net reward for this coalition
            net_reward = phase2_utils.calculate_net_reward(robot_team, task)
            
            # Update best net reward if current is better
            if net_reward > best_net_reward:
                best_net_reward = net_reward
                best_team = robot_team
    
    # Calculate resource utilization and distance penalty for robots not in best team
    potential_resource_utilization = 0
    capabilities = np.zeros(kappa)
    distance_penalty = 0
    max_resources = task.get_max_resources()
    
    for robot in robots:
            
            capabilities += robot.get_capabilities()
            
            # Calculate distance penalty -is this average distance?
            distance_penalty += np.linalg.norm(np.array(robot.get_location()) - np.array(task.get_location()))
            
    for i in range(kappa):
        potential_resource_utilization += min(max_resources[i],capabilities[i])
        
    # # Calculate final coalition value
    # print("\n Robots, Tasks: ", len(robots), len(tasks))
    # print("\nbest_net_reward: ", best_net_reward)
    # print("unused_cap: ", unused_cap)
    # print("potential_resource_utilization: ", potential_resource_utilization)
    # print("distance penalty: ", distance_penalty)  
    coalition_val = 1000*best_net_reward + 100*potential_resource_utilization - distance_penalty
    #print("coalition_val ", coalition_val)  
    return coalition_val


# Here choose which value function to use
def coalition_value(robots,tasks,kappa):
    return coalition_value_3(robots,tasks,kappa)
    

# Shapley cap matching, task overlap, avg distance
# Result - Tasks grouped together with exact same type of tasks    
def coalition_value_1(robots, tasks, kappa):
    # Coalition value function 1: 
    # Shapley value based capability matching + task overlap - avg distance to tasks

    # Define weights for each term
    alpha_1 = 1 # Capability matiching term
    alpha_2 = 10 # Task overlap term
    alpha_3 = 1 # Average distance to tasks term

    cap_reward = shapley_capability_matching(robots,tasks,kappa)
    task_overlap_rew = task_overlap(tasks,kappa)
    est_distance = avg_distance_cost(robots,tasks)

    coalition_value = alpha_1*cap_reward + alpha_2*task_overlap_rew - alpha_3*est_distance

    return coalition_value
    """
      Capability matching term ------------------------
    """



    # m = len(tasks)
    # n = len(robots)
    
    # # Create shapley vectors
    # shapley_vectors = [np.array([]) for _ in range(kappa)]
    # for task in tasks:
    #     task_shapleys = task.get_grand_shapley_vectors()
    #     for i in range(kappa):
    #         shapley_vectors[i] = np.concatenate([shapley_vectors[i],task_shapleys[i]])
    
    # # Sort shapley vectors in descending order
    # for i in range(kappa):
    #     shapley_vectors[i] = np.sort(shapley_vectors[i])[::-1]

    # # Print shapley vectors
    # #print("Shapley vectors: ", shapley_vectors)
    
    # coalition_capabilities = np.zeros(kappa, dtype=int)
    # for robot in robots:
    #     coalition_capabilities += robot.get_capabilities().astype(int)
    
    # # Calculate the total reward from capability matching
    # capability_reward = 0
    # #print("Coalition capabilities: ", coalition_capabilities)
    # for capability in range(kappa):
    #     # For each instance of the capability, add the shapley value
    #     for j in range(min(coalition_capabilities[capability], len(shapley_vectors[capability]))):
    #         capability_reward += shapley_vectors[capability][j]
    
    # #print("Capability reward: ", capability_reward)

    
    # """
    # # Task overlap term --------------------------------
    # """
    # task_overlap = 0
    # for i in range(m):
    #     for j in range(i+1,m):
    #         grand_coalition_i = tasks[i].get_grand_coalition()
    #         grand_coalition_j = tasks[j].get_grand_coalition()
    #         for k in range(kappa):
    #             task_overlap += min(grand_coalition_i[k], grand_coalition_j[k])

    # #print("Task overlap: ", task_overlap)


    # """
    # # Average distance to tasks term -------------------
    # """
    # est_cost = 0
    # for robot in robots:
    #     for task in tasks:
    #         est_cost += (1/m) * np.linalg.norm(np.array(robot.get_location()) - np.array(task.get_location())) 
    
    # #print("Estimated cost: ", est_cost)

    # coalition_value = alpha_1 * capability_reward + alpha_2 * task_overlap - alpha_3 * est_cost

    # #print("Coalition value: ", coalition_value)

    # return coalition_value

# Shapley cap matching, num tasks, avg distance
# Result - tasks bunched up in groups without robots
def coalition_value_2(robots, tasks, kappa):
    # Coalition value function 2: 
    # Shapley value based capability matching + task overlap - avg distance to tasks

    # Define weights for each term
    alpha_1 = 1 # Capability matiching term
    alpha_2 = 50 # Number of tasks term
    alpha_3 = 1 # Average distance to tasks term

    
    """
      Capability matching term ------------------------
    """

    m = len(tasks)
    n = len(robots)
    
    # Create shapley vectors
    shapley_vectors = [np.array([]) for _ in range(kappa)]
    for task in tasks:
        task_shapleys = task.get_grand_shapley_vectors()
        for i in range(kappa):
            shapley_vectors[i] = np.concatenate([shapley_vectors[i],task_shapleys[i]])
    
    # Sort shapley vectors in descending order
    for i in range(kappa):
        shapley_vectors[i] = np.sort(shapley_vectors[i])[::-1]

    # Print shapley vectors
    #print("Shapley vectors: ", shapley_vectors)
    
    coalition_capabilities = np.zeros(kappa, dtype=int)
    for robot in robots:
        coalition_capabilities += robot.get_capabilities().astype(int)
    
    # Calculate the total reward from capability matching
    capability_reward = 0
    #print("Coalition capabilities: ", coalition_capabilities)
    for capability in range(kappa):
        # For each instance of the capability, add the shapley value
        for j in range(min(coalition_capabilities[capability], len(shapley_vectors[capability]))):
            capability_reward += shapley_vectors[capability][j]
    
    #print("Capability reward: ", capability_reward)

    
    """
    # Num tasks term --------------------------------
    """
    additional_tasks = len(tasks)-1
    #print("tasks: ", tasks)
    #print("tasks_reward: ", additional_tasks*alpha_2)
    # task_overlap = 0
    # for i in range(m):
    #     for j in range(i+1,m):
    #         grand_coalition_i = tasks[i].get_grand_coalition()
    #         grand_coalition_j = tasks[j].get_grand_coalition()
    #         for k in range(kappa):
    #             task_overlap += min(grand_coalition_i[k], grand_coalition_j[k])

    #print("Task overlap: ", task_overlap)


    """
    # Average distance to tasks term -------------------
    """
    est_cost = 0
    for robot in robots:
        for task in tasks:
            est_cost += (1/m) * np.linalg.norm(np.array(robot.get_location()) - np.array(task.get_location())) 
    
    #print("Estimated cost: ", est_cost)

    coalition_value = alpha_1 * capability_reward + alpha_2 * additional_tasks - alpha_3 * est_cost

    #print("Coalition value: ", coalition_value)

    return coalition_value

# Shapley cap matching, flexibility, avg distance
def coalition_value_3(robots,tasks,kappa):
    # Define weights for each term
    alpha_1 = 1000 # Capability matching term
    alpha_2 = 100 # Flexibility term
    alpha_3 = 1 # Average distance to tasks term

    cap_reward = shapley_capability_matching(robots,tasks,kappa)
    flex_reward = flexibility_reward(robots,tasks,kappa)
    est_distance = avg_distance_cost(robots,tasks)

    coalition_value = alpha_1*cap_reward + alpha_2*flex_reward - alpha_3*est_distance

    return coalition_value

# LB cap matching, flexibility, avg distance
def coalition_value_4(robots,tasks,kappa):
    # Define weights for each term
    alpha_1 = 1000 # Capability matching term
    alpha_2 = 100 # Flexibility term
    alpha_3 = 1 # Average distance to tasks term

    cap_reward = lower_bound_cap_matching(robots,tasks,kappa,print_out=False)
    flex_reward = flexibility_reward(robots,tasks,kappa)
    est_distance = avg_distance_cost(robots,tasks)

    coalition_value = alpha_1*cap_reward + alpha_2*flex_reward - alpha_3*est_distance

    return coalition_value