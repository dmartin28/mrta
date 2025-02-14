# Phase 2 assignment algorithm
import numpy as np
from task import Task
from robot import Robot

nu = 10 #number of robots
mu = 5 # number of tasks
kappa = 2 # number of capabilities
L = 3 # maximum team size

#Define the environment size
max_x = 10
min_x = 0
max_y = 10
min_y = 0

####### Define some specific task types: ############
reward_dim = tuple(L+1 for _ in range(kappa))

#Type 1 can be done by robots with capability 1 
task_type_1 = np.zeros(reward_dim)
task_type_1[1,0] = 100
task_type_1[2,0] = 200

#Type 2 can be done by robots with capability 2 
task_type_2 = np.zeros(reward_dim)
task_type_2[0,1] = 100
task_type_2[0,2] = 200
task_type_2[0,3] = 250

#Type 3 can be done only collaboratively by cap 1 and 2 
task_type_3 = np.zeros(reward_dim)
task_type_3[1,1] = 200
task_type_3[1,2] = 220
task_type_3[2,1] = 220

#Type 4 can be done only collaboratively by two of cap 1 
task_type_4 = np.zeros(reward_dim)
task_type_4[2,0] = 200

#Type 5 can be done only collaboratively by two of cap 2 
task_type_5 = np.zeros(reward_dim)
task_type_5[0,2] = 200
#####################################################

#Define the two robot types:

robot_type_1 = [1,0]
robot_type_2 = [0,1]

#Generate random robot and task locations
robot_x_locations = np.random.uniform(min_x, max_x, nu)
robot_y_locations = np.random.uniform(min_y, max_y, nu)
task_x_locations = np.random.uniform(min_x, max_x, mu)
task_y_locations = np.random.uniform(min_y, max_y, mu)

robot_list = []
task_list = []

# Create robots
for i in range(nu):
    if i < nu // 2:
        robot_type = robot_type_1
    else:
        robot_type = robot_type_2
    
    robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
    robot_list.append(robot)

# Create tasks
task_types = [task_type_1, task_type_2, task_type_3, task_type_4, task_type_5]
for i in range(mu):
    task = Task(i, task_types[i], task_x_locations[i], task_y_locations[i])
    task_list.append(task)

Assignment = IP_assignment(robot_list, task_list, L)


# print(f"Created {len(robot_list)} robots:")
# for robot in robot_list:
#     print(f"Robot {robot.id}: Type {robot.capabilities}, Position ({robot.x}, {robot.y})")

# print(f"\nCreated {len(task_list)} tasks:")
# for task in task_list:
#     print(f"Task {task.id}: Reward: {task.reward_matrix} Position ({task.x}, {task.y})")