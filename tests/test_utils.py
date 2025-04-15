import numpy as np
import random
from shared_classes.task import Task
from shared_classes.robot import Robot

def generate_task_types(L, kappa):
    # Reward matrix dimensions is (L+1)^kappa (0 to L for each capability)
    reward_dim = tuple(L+1 for _ in range(kappa))

    task_types = {
        0: np.zeros(reward_dim),  # Type 0: can be done by robots with capability 1
        1: np.zeros(reward_dim),  # Type 1: can be done by robots with capability 2
        2: np.zeros(reward_dim),  # Type 2: can be done only collaboratively by cap 1 and 2
        3: np.zeros(reward_dim),  # Type 3: can be done only collaboratively by two of cap 1
        4: np.zeros(reward_dim),  # Type 4: can be done only collaboratively by two of cap 2
        5: np.zeros(reward_dim),  # Type 5: can be done only collaboratively by cap 1 and 2, diminishing returns
        6: np.zeros(reward_dim),  # Type 6: can be done only individually by type 1
        7: np.zeros(reward_dim),  # Type 7: can be done only individually by type 2
        8: np.zeros(reward_dim),  # Type 8: can be done by either type, only individually
        9: np.zeros(reward_dim),  # Type 9: can be done by either type but needs 3 robots
    }

    # Type 0
    task_types[0][1,0] = 100
    task_types[0][2,0] = 200

    # Type 1
    task_types[1][0,1] = 100
    task_types[1][0,2] = 150
    task_types[1][0,3] = 200

    # Type 2
    task_types[2][1,1] = 200
    task_types[2][1,2] = 300
    task_types[2][2,1] = 300

    # Type 3
    task_types[3][2,0] = 200

    # Type 4
    task_types[4][0,2] = 200

    # Type 5
    task_types[5][1,1] = 200
    task_types[5][1,2] = 220
    task_types[5][2,1] = 220

    # Type 6
    task_types[6][1,0] = 100

    # Type 7
    task_types[7][0,1] = 100

    # Type 8
    task_types[8][1,0] = 100
    task_types[8][0,1] = 100

    # Type 9
    task_types[9][1,2] = 350
    task_types[9][2,1] = 350
    task_types[9][3,0] = 350
    task_types[9][0,3] = 350

    return task_types

def generate_problem_instance(hypes, max_x, max_y):
    
    robot_type_1 = [1,0]
    robot_type_2 = [0,1]
    min_x = 100
    min_y = 100
    nu = hypes['nu']  # number of robots
    mu = hypes['mu']  # number of tasks
    kappa = hypes['kappa']  # number of capabilities
    L = hypes['L']  # maximum team size

    # Generate random robot and task locations
    robot_x_locations = np.round(np.random.uniform(min_x, max_x, nu), decimals=1)
    robot_y_locations = np.round(np.random.uniform(min_y, max_y, nu), decimals=1)
    task_x_locations = np.round(np.random.uniform(min_x, max_x, mu), decimals=1)
    task_y_locations = np.round(np.random.uniform(min_y, max_y, mu), decimals=1)

    robot_list = []
    task_list = []

    # Create robots
    for i in range(nu):
        robot_type =random.choice([robot_type_1, robot_type_2])       
        robot = Robot(i,robot_type, robot_x_locations[i], robot_y_locations[i])
        robot_list.append(robot)

    # Create tasks
    task_types = generate_task_types(L, kappa)
    task_type_keys = list(task_types.keys())
    for i in range(mu):
        task_type_key = random.choice(task_type_keys)
        task_type = task_types[task_type_key]
        task = Task(i, task_type, task_x_locations[i], task_y_locations[i])
        task_list.append(task)

    return robot_list, task_list

def print_problem_instance(robot_list, task_list):
    print(f"Created {len(robot_list)} robots:")
    for robot in robot_list:
        print(f"Robot {robot.id}: Position ({robot.x}, {robot.y})")
        print(f"Capabilities: {robot.capabilities}")
        print()  # Add an empty line for better readability

    print(f"\nCreated {len(task_list)} tasks:")
    for task in task_list:
        print(f"Task {task.id}: Position ({task.x}, {task.y})")
        print(f"Reward Matrix:\n{task.reward_matrix}")
        print()  # Add an empty line for better readability