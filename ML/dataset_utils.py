import random

""" This code returns a pair of cluster sizes """
def generate_cluster_sizes(L_r, L_t):
    """
    Generate random cluster sizes for a pair of clusters.
    
    Parameters:
    L_r (int): Maximum number of robots in a cluster.
    L_t (int): Maximum number of tasks in a cluster.
    
    Returns:
    tuple: A tuple containing the number of tasks and robots in the first cluster, 
           and the number of tasks and robots in the second cluster.
    """
    while True:
        # Generate random number of tasks for each cluster
        num_tasks_1 = random.randint(0, L_t)
        num_tasks_2 = random.randint(0, L_t)

        # Generate random number of robots for each cluster
        num_robots_1 = random.randint(0, L_r)
        num_robots_2 = random.randint(0, L_r)

        # Check if the distribution is valid
        if (num_tasks_1 > 0 or num_robots_1 > 0) and (num_tasks_2 > 0 or num_robots_2 > 0):
            if (num_tasks_1 + num_tasks_2 <= L_t) and (num_robots_1 + num_robots_2 <= L_r):
                break

    return (num_tasks_1, num_robots_1), (num_tasks_2, num_robots_2)