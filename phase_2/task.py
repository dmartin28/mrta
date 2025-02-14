import numpy as np

class Task:
    def __init__(self, id, reward_matrix, x, y):
        self.reward_matrix = np.array(reward_matrix)
        self.x = x
        self.y = y
        self.id = id

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

    def __str__(self):
        return f"Task with reward matrix:\n{self.reward_matrix}"