def convert_assignment_to_clusters(assignment):
    """
    Convert the assignment to clusters where  each clster contains either:
        1. A task and its assigned robots
        2. A task only
        3. A robot only"""
    
    # Clusters is a list of clusters
    # Each cluster is 2D array, row 1 are robot indices, row 2 are task ids
    clusters = []

    # First list of robot ids in assignments is a dummy task that represents unassigned robots
    # We need to create a cluster for each robot in the dummy task
    unassigned_robots = assignment[0]
    for robot_id in unassigned_robots:
        # Create a cluster with just the robot
        clusters.append([[robot_id], []])

    # For all other tasks in the assignment list we must create a cluster containing 
    # the task and its assigned robots
    for task_idx in range(1, len(assignment)):
        
        # Get the robots assigned to the task
        robots = assignment[task_idx]
        
        # Since the first task is a dummy task, the true task id is task_idx - 1
        task_id = task_idx - 1
        
        # If there are no robots assigned to the task, create a cluster with just the task
        if len(robots) == 0:
            clusters.append([[], [task_id]])
        else:
            # Create a cluster with the task and its assigned robots
            clusters.append([robots, [task_id]])

    return clusters

def convert_assignment_to_clusters_new(assignment):
    """
    Convert the assignment to clusters where each cluster contains either:
        1. A task and its assigned robots
        2. A task only
        3. A robot only

    Args:
    assignment (dict): A dictionary where keys are task IDs (and -1 for unassigned),
                       and values are lists of robot IDs assigned to each task.

    Returns:
    list: A list of clusters, where each cluster is a 2D array:
          row 1 contains robot IDs, row 2 contains task IDs (empty for unassigned robots)
    """
    
    # Clusters is a list of clusters
    # Each cluster is 2D array, row 1 are robot IDs, row 2 are task IDs
    clusters = []

    # Handle unassigned robots (key -1)
    unassigned_robots = assignment.get(-1, [])
    for robot_id in unassigned_robots:
        # Create a cluster with just the robot
        clusters.append([[robot_id], []])

    # Handle assigned tasks and robots
    for task_id, robots in assignment.items():
        if task_id != -1:  # Skip the unassigned robots key
            if len(robots) == 0:
                # If there are no robots assigned to the task, create a cluster with just the task
                clusters.append([[], [task_id]])
            else:
                # Create a cluster with the task and its assigned robots
                clusters.append([robots, [task_id]])
                
    return clusters