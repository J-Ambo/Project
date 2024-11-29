import numpy as np

class DataRecorder:
    def __init__(self, population, time, repetitions):
        self.all_agents = population.population_array
        self.population = population

        self.data = np.zeros((repetitions, time, len(self.all_agents), 2, self.population.dimension))
        self.polarisation_data = np.zeros((repetitions, time))
        self.rotation_data = np.zeros((repetitions, time))

        self.data[0][0][:, 0] = [agent.position for agent in self.all_agents]
        self.data[0][0][:, 1] = [agent.direction for agent in self.all_agents]
             
    def update_data(self, time, repetitions):
        self.data[repetitions][time][:, 0] = [agent.position for agent in self.all_agents]
        self.data[repetitions][time][:, 1] = [agent.direction for agent in self.all_agents]
        self.polarisation_data[repetitions][time] = self.population.polarisation
        self.rotation_data[repetitions][time] = self.population.rotation
        
    def get_data(self):
        return self.data, self.polarisation_data, self.rotation_data
