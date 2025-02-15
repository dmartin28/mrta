def IP_assignment(robot_list, task_list, L):
    
    n = len(robot_list)
    m = len(task_list)

    # Assignment is a 2D ragged list first index corresponds to tasks,
    # second is index of robot in robotlist.
    assignment = [[]for _ in range(m)]
    for i in range(0,n):
        assignment[robot_list[i] % m].append(robot_idx)
    return assignment

def IP_recursion(robot_list, task_list, task_max_vals L):
    n = len(robot_list)
    m = len(task_list)
    assignment = [[]for _ in range(m)]
    for robot in robot_list:
        assignment[robot.id % m].append(robot.id)
    return assignment