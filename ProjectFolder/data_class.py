import numpy as np

class DataRecorder:
    def __init__(self, population_size, dimensions, time, repetitions):
        self.data = np.zeros((repetitions, time, population_size, 2, dimensions))
        self.polarisation_data = np.zeros((repetitions, time))
        self.rotation_data = np.zeros((repetitions, time))

    def update_data(self, population, time, repetitions):
        self.all_agents = population.population_array
        self.population = population
        self.data[repetitions][time][:, 0] = [agent.position for agent in self.all_agents]
        self.data[repetitions][time][:, 1] = [agent.direction for agent in self.all_agents]
        self.polarisation_data[repetitions][time] = self.population.polarisation
        self.rotation_data[repetitions][time] = self.population.rotation
        
    def get_data(self):
        return self.data, self.polarisation_data, self.rotation_data

'''

from population_class import Population
from environment_class import Environment
env = Environment(10, 3)
pop = Population(4, 2, env)
data = DataRecorder(pop, 5, 2)
print(data.get_data())
'''