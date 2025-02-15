def IP_assignment(robot_list, task_list, L):
    
    n = len(robot_list)
    m = len(task_list)


    # C
    # assignment is a 2D ragged list First index corresponds to tasks.
    assignment = [[]for _ in range(m)]
    for robot in robot_list:
        assignment[robot.id % m].append(robot.id)
    return assignment

def IP_recursion(robot_list, task_list, task_max_vals L):
    n = len(robot_list)
    m = len(task_list)
    assignment = [[]for _ in range(m)]
    for robot in robot_list:
        assignment[robot.id % m].append(robot.id)
    return assignment