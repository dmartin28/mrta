import numpy as np
from itertools import product
import math

class Task:
    def __init__(self, id, reward_matrix, x, y):
        self.reward_matrix = np.array(reward_matrix)
        self.x = x
        self.y = y
        self.id = id
        
        # Determine grand coalition for task
        flat_index = np.argmax(self.reward_matrix)
        grand_coalition_tuple = np.unravel_index(flat_index, self.reward_matrix.shape)
        self.grand_coalition = np.array(grand_coalition_tuple)
        self.grand_shapley_vectors = self.get_shapley_vectors(self.grand_coalition)


    def get_id(self):
        return self.id
    
    def get_grand_coalition(self):
        return self.grand_coalition

    def get_reward_matrix(self):
        return self.reward_matrix

    def get_dimensions(self):
        return self.reward_matrix.shape

    def get_reward(self, *indices):
        try:
            return self.reward_matrix[indices]
        except IndexError:
            raise IndexError("Indices out of bounds for the reward matrix.")

    def set_reward(self, value, *indices):
        try:
            self.reward_matrix[indices] = value
        except IndexError:
            raise IndexError("Indices out of bounds for the reward matrix.")

    def get_location(self):
        return (self.x, self.y)
    
    def set_location(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Task with reward matrix:\n{self.reward_matrix}"
    
    def get_shapley_vectors(self,coalition): # Note: coalition is an np array of capability amounts
        """
        Calculate the Shapley value for a task
        """

        #print("Grand Coalition: ", coalition)
    
        n = np.sum(coalition) # number of robots
        kappa = len(coalition) # number of capabilities
        shapley_values = np.zeros(kappa)

        for capability in range(kappa):
            # create subsets that do not include the capability
            player = np.zeros(kappa)
            player[capability] = 1
            teamate_capabilities = (coalition.copy() - player).astype(int)
        
            # Create ranges for each element
            ranges = [range(i+1) for i in teamate_capabilities]

            # Use itertools.product to generate all combinations
            tuples = list(product(*ranges))

            # Convert tuples to numpy arrays
            subsets = [np.array(subset) for subset in tuples]

            #print("Capability: ", capability)
            #print("subsets of teamate capabilities: ", subsets)

            for subset in subsets:
                #print("subset: ", subset)
                #print("subset+player: ", subset+player)

                # Calculate number of times each subset occurs
                subset_multiplicity = 1
                for k in range(kappa):
                    subset_multiplicity *= math.comb(teamate_capabilities[k], subset[k])
                #print("subset_multiplicity: ", subset_multiplicity)

                MC = self.get_reward(*((subset+player).astype(int))) - self.get_reward(*subset)
                shapley_values[capability] += MC/(math.comb((n-1), np.sum(subset))) * subset_multiplicity
            shapley_values[capability] = shapley_values[capability] / n

        # Create shapley vector as ragged array
        shapley_vectors = [np.full(coalition[k], shapley_values[k]) for k in range(kappa)]
    
        #print("Shapley vectors: ", shapley_vectors)

        return shapley_vectors
    
    def get_grand_shapley_vectors(self):
        return self.grand_shapley_vectors