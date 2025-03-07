import matplotlib.pyplot as plt
import numpy as np

def plot_clusters(clusters, assignments, robot_list, task_list, max_x, max_y, ax=None, title='Robot-Task Clusters and Assignments'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Generate a list of distinct colors for each cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    
    for i, (cluster, color) in enumerate(zip(clusters, colors)):
        robots = cluster[0]
        tasks = cluster[1]
        
        # Plot robots as circles (type 1) or triangles (type 2)
        for robot_id in robots:
            robot = robot_list[robot_id]
            if robot.capabilities[0] == 1:  # Type 1 robot
                marker = 'o'
            else:  # Type 2 robot
                marker = '^'
            ax.plot(robot.x, robot.y, marker, color=color, markersize=10, label=f'Robot {robot_id}')
        
        # Plot tasks as X's
        for task_id in tasks:
            task = task_list[task_id]
            ax.plot(task.x, task.y, 'x', color=color, markersize=10, label=f'Task {task_id}')
        
        # Draw lines between assigned robots and tasks
        for task_idx, assigned_robots in enumerate(assignments[i][1:]):
            task = task_list[cluster[1][task_idx]]
            for robot_id in assigned_robots:
                robot = robot_list[robot_id]
                ax.plot([robot.x, task.x], [robot.y, task.y], '--', color=color, alpha=0.5)
    
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title)
    
    # Create legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add a legend for robot types
    ax.plot([], [], 'ko', label='Type 1 Robot')
    ax.plot([], [], 'k^', label='Type 2 Robot')
    ax.plot([], [], 'kx', label='Task')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return ax
    
def plot_assignments(assignment, robot_list, task_list, max_x, max_y, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot robots
    for robot in robot_list:
        if robot.capabilities[0] == 1:  # Type 1 robot
            marker = 'o'
        else:  # Type 2 robot
            marker = '^'
        ax.plot(robot.x, robot.y, marker, color='blue', markersize=10, label=f'Robot {robot.id}')
    
    # Plot tasks
    for task in task_list:
        ax.plot(task.x, task.y, 'x', color='red', markersize=10, label=f'Task {task.id}')
    
    # Draw lines between assigned robots and tasks
    for task_id, assigned_robots in enumerate(assignment[1:]):
        task = task_list[task_id]
        for robot_id in assigned_robots:
            robot = robot_list[robot_id]
            ax.plot([robot.x, task.x], [robot.y, task.y], '--', color='green', alpha=0.5)
    
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(title)
    
    # Create legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add a legend for robot types and tasks
    ax.plot([], [], 'bo', label='Type 1 Robot')
    ax.plot([], [], 'b^', label='Type 2 Robot')
    ax.plot([], [], 'rx', label='Task')
    ax.plot([], [], 'g--', label='Assignment')
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    return ax