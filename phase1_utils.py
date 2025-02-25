import numpy as np
from itertools import product
import math

from task import Task
from robot import Robot

# def shapley_vector(task):
#     """
#     Calculate the Shapley value for a task
#     """

#     """
#     # Note this should likely be moved inside of the Task class
#     """

#     # Find the grand coalition capabilities:
#     # reward_matrix = task.get_reward_matrix()
#     # flat_index = np.argmax(reward_matrix)
#     # grand_coalition_tuple = np.unravel_index(flat_index, reward_matrix.shape)
#     # grand_coalition = np.array(grand_coalition_tuple)

#     grand_coalition = task.get_grand_coalition()

#     print("Grand Coalition: ", grand_coalition)
    
#     n = np.sum(grand_coalition) # number of robots
#     kappa = len(grand_coalition) # number of capabilities
#     shapley_values = np.zeros(kappa)

#     for capability in range(kappa):
#         # create subsets that do not include the capability
#         player = np.zeros(kappa)
#         player[capability] = 1
#         teamate_capabilities = (grand_coalition.copy() - player).astype(int)
        
#         # Create ranges for each element
#         ranges = [range(i+1) for i in teamate_capabilities]

#         # Use itertools.product to generate all combinations
#         tuples = list(product(*ranges))

#         # Convert tuples to numpy arrays
#         subsets = [np.array(subset) for subset in tuples]

#         print("Capability: ", capability)
#         print("subsets of teamate capabilities: ", subsets)

#         for subset in subsets:
#             print("subset: ", subset)
#             print("subset+player: ", subset+player)

#             # Calculate number of times each subset occurs
#             subset_multiplicity = 1
#             for k in range(kappa):
#                 subset_multiplicity *= math.comb(teamate_capabilities[k], subset[k])
#             print("subset_multiplicity: ", subset_multiplicity)

#             MC = task.get_reward(*((subset+player).astype(int))) - task.get_reward(*subset)
#             shapley_values[capability] += MC/(math.comb((n-1), np.sum(subset))) * subset_multiplicity
#         shapley_values[capability] = shapley_values[capability] / n

#     # Create shapley vector as ragged array
#     shapley_vectors = [np.full(grand_coalition[k], shapley_values[k]) for k in range(kappa)]
    
#     print("Shapley vectors: ", shapley_vectors)

#     return shapley_vectors
    
def coalition_value_1(robots, tasks, kappa):
    # Coalition value function 1: 
    # Shapley value based capability matching + task overlap - avg distance to tasks

    # Define weights for each term
    alpha_1 = 1 # Capability matiching term
    alpha_2 = 10 # Task overlap term
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
    # Task overlap term --------------------------------
    """
    task_overlap = 0
    for i in range(m):
        for j in range(i+1,m):
            grand_coalition_i = tasks[i].get_grand_coalition()
            grand_coalition_j = tasks[j].get_grand_coalition()
            for k in range(kappa):
                task_overlap += min(grand_coalition_i[k], grand_coalition_j[k])

    #print("Task overlap: ", task_overlap)


    """
    # Average distance to tasks term -------------------
    """
    est_cost = 0
    for robot in robots:
        for task in tasks:
            est_cost += (1/m) * np.linalg.norm(np.array(robot.get_location()) - np.array(task.get_location())) 
    
    #print("Estimated cost: ", est_cost)

    coalition_value = alpha_1 * capability_reward + alpha_2 * task_overlap - alpha_3 * est_cost

    #print("Coalition value: ", coalition_value)

    return coalition_value

