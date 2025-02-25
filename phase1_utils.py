from task import Task
import numpy as np
from itertools import product
import math

def shapley_vector(task):
    """
    Calculate the Shapley value for a task
    """

    # Find the grand coalition capabilities:
    reward_matrix = task.get_reward_matrix()
    flat_index = np.argmax(reward_matrix)
    grand_coalition_tuple = np.unravel_index(flat_index, reward_matrix.shape)
    grand_coaltion = grand_coalition = np.array(grand_coalition_tuple)

    print("Grand Coalition: ", grand_coalition)
    
    n = np.sum(grand_coalition) # number of robots
    kappa = len(grand_coalition) # number of capabilities
    shapley_values = np.zeros(kappa)

    for capability in range(kappa):
        # create subsets that do not include the capability
        player = np.zeros(kappa)
        player[capability] = 1
        teamate_capabilities = (grand_coalition.copy() - player).astype(int)
        
        # Create ranges for each element
        ranges = [range(i+1) for i in teamate_capabilities]

        # Use itertools.product to generate all combinations
        tuples = list(product(*ranges))

        # Convert tuples to numpy arrays
        subsets = [np.array(subset) for subset in tuples]

        print("Capability: ", capability)
        print("subsets of teamate capabilities: ", subsets)

        for subset in subsets:
            print("subset: ", subset)
            print("subset+player: ", subset+player)
            MC = task.get_reward(*((subset+player).astype(int))) - task.get_reward(*subset)
            shapley_values[capability] += MC/(math.comb((n-1), np.sum(subset)))
        shapley_values[capability] = shapley_values[capability] / n

    return shapley_values
    