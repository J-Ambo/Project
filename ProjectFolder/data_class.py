import numpy as np
from line_profiler import profile

class DataRecorder:
    def __init__(self, population_size, dimensions, time, repetitions, increments):
        #self.data = np.zeros((increments, repetitions, time, population_size, 2, dimensions))
        self.position_data = np.zeros((increments, repetitions, time, population_size, dimensions))
        self.polarisation_data = self._initialize_data(increments, repetitions, time)
        self.rotation_data = self._initialize_data(increments, repetitions, time)
    
    def _initialize_data(self, increments, repetitions, time):
        return np.array([[[np.zeros(time), np.zeros(8)] for _ in range(repetitions)] for _ in range(increments)], dtype=object)

    def update_parameters(self, increment, repetition, parameters):
        self.polarisation_data[increment][repetition][1] = parameters 
        self.rotation_data[increment][repetition][1] = parameters

    def update_data(self, population, increment, repetition, time):
        self.all_agents = population.population_array
        self.population = population
        #self.data[increment, repetition, time, :, 0, :] = self.population.population_positions           #np.array([agent.position for agent in self.all_agents])
        #self.data[increment, repetition, time, :, 1, :] = self.population.population_directions          #np.array([agent.direction for agent in self.all_agents])

        self.position_data[increment, repetition, time, :] = self.population.population_positions
        self.polarisation_data[increment][repetition][0][time] = self.population.polarisation
        self.rotation_data[increment][repetition][0][time] = self.population.rotation

    def get_polarisation_data(self):
        return self.polarisation_data
    
    def get_rotation_data(self):
        return self.rotation_data
    
    def get_position_data(self):
        return self.position_data
    
data = DataRecorder(5, 3, 6, 1, 2)


#print(data.get_data()[2])
#print(data.get_data()[2][0])
#print(data.get_data()[2][0,0])
#print(data.get_data()[2][0,0,0])
#print(data.get_data()[2][0,0,0][0])