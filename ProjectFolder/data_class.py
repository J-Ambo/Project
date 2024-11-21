import numpy as np

class DataRecorder:
    def __init__(self, all_agents):
        self.all_agents = all_agents
        self.data = np.zeros((len(all_agents), 3, 2))
        self.data[:, 0] = [agent.position for agent in all_agents]
        self.data[:, 1] = [agent.direction for agent in all_agents]
        self.data[:, 2] = [agent.speed for agent in all_agents]
             
    def update_data(self):
        self.data[:, 0] = [agent.position for agent in self.all_agents]
        self.data[:, 1] = [agent.direction for agent in self.all_agents]
        self.data[:, 2] = [agent.speed for agent in self.all_agents]
        
    def get_data(self):
        return self.data
    
agents_posn = [np.array([1,1]), np.array([1,2]), np.array([0,1])]
lst = [0, 2]
print(sum(np.array([0,0]) - agents_posn[n] for n in lst))

print((np.array([1,1]), np.array([2,1]) ))