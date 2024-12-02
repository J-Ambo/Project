import numpy as np

class DataRecorder:
    def __init__(self, population_size, dimensions, time, repetitions):
        self.data = np.zeros((repetitions, time, population_size, 2, dimensions))
        self.polarisation_data = np.array([[np.zeros(time), np.zeros(7)] for _ in range(repetitions)], dtype=object)
        self.rotation_data = np.array([[np.zeros(time), np.zeros(7)] for _ in range(repetitions)], dtype=object)
    
    def update_parameters(self, repetitions, parameters):
        self.polarisation_data[repetitions][1] = parameters 
        self.rotation_data[repetitions][1] = parameters

    def update_data(self, population, time, repetitions):
        self.all_agents = population.population_array
        self.population = population
        self.data[repetitions][time][:, 0] = [agent.position for agent in self.all_agents]
        self.data[repetitions][time][:, 1] = [agent.direction for agent in self.all_agents]
        self.polarisation_data[repetitions][0][time] = self.population.polarisation
        self.rotation_data[repetitions][0][time] = self.population.rotation
        
    def get_data(self):
        return self.data, self.polarisation_data, self.rotation_data
