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

#Type 1 can be done by robots with capability 1 
task_type_1 = np.zeros(reward_dim)
task_type_1[1,0] = 100
task_type_1[2,0] = 200

#Type 2 can be done by robots with capability 2 
task_type_2 = np.zeros(reward_dim)
task_type_2[0,1] = 100
task_type_2[0,2] = 150
task_type_2[0,3] = 200

#Type 3 can be done only collaboratively by cap 1 and 2 
task_type_3 = np.zeros(reward_dim)
task_type_3[1,1] = 200
task_type_3[1,2] = 300
task_type_3[2,1] = 300

#Type 4 can be done only collaboratively by two of cap 1 
task_type_4 = np.zeros(reward_dim)
task_type_4[2,0] = 200

#Type 5 can be done only collaboratively by two of cap 2 
task_type_5 = np.zeros(reward_dim)
task_type_5[0,2] = 200

#Type 6 can be done only collaboratively by cap 1 and 2 
task_type_6 = np.zeros(reward_dim)
task_type_6[1,1] = 200
task_type_6[1,2] = 220
task_type_6[2,1] = 220

# #Additional shapley tests
# task_type_6[1,0] = 50
# task_type_6[2,1] = 200
# task_type_6[2,2] = 300

#Type 7 can be done only collaboratively by two of cap 1 
task_type_7 = np.zeros(reward_dim)
task_type_7[1,0] = 100

#Type 8 can be done only collaboratively by two of cap 2 
task_type_8 = np.zeros(reward_dim)
task_type_8[0,1] = 100
#####################################################

task = Task(0, task_type_6, 0, 0)

#utils_shapley_values = utils.shapley_vector(task)

task_shapley_values = task.get_grand_shapley_vector()

#print("Utils Shapley Values: ", utils_shapley_values)
print("Task Shapley Values: ", task_shapley_values)