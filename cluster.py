

class Robot:
    def __init__(self, id, robots, tasks):
        
        self.robots = robots
        self.tasks = tasks
        self.id = id

    def get_id(self):
        return self.id 
    
    def get_robots(self):
        return self.robots
    
    def get_tasks(self):
        return self.tasks
    
    def add_robot(self, robot):
        self.robots.append(robot)

    def add_task(self, task):
        self.tasks.append(task)
    
    def remove_robot(self, robot):
        self.robots.remove(robot)