import numpy as np
from task import Task
from robot import Robot
import phase1_utils as utils

nu = 11 #number of robots
mu = 5 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size

#Define the environment size
max_x = 10
min_x = 0
max_y = 10
min_y = 0

####### Define some specific task types: ############

#Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
reward_dim = tuple(L+1 for _ in range(kappa))

#Type 3 can be done only collaboratively by cap 1 and 2 
task_type_3 = np.zeros(reward_dim)
task_type_3[1,1] = 200
task_type_3[1,2] = 250
task_type_3[2,1] = 300

task = Task(0, task_type_3, 0, 0)

shapley_values = utils.shapley_vector(task)

print("Shapley Values: ", shapley_values)