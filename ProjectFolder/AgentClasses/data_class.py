import numpy as np
from line_profiler import profile

class DataRecorder:
    def __init__(self, population_size, dimensions, time, repetitions, increments, strips):
        self.position_data = np.zeros((strips, increments, repetitions, time, population_size, dimensions))
        self.direction_data = np.zeros((strips, increments, repetitions, time, population_size, dimensions))
        self.density_data = np.zeros((strips, increments, repetitions, time, population_size))

        self.predator_position_data = np.zeros((strips, increments, repetitions, time, 1, dimensions))
        self.predator_direction_data = np.zeros((strips, increments, repetitions, time, 1, dimensions))
        self.predator_prey_distances = np.zeros((strips, increments, repetitions, time))
        self.predator_attack_number = np.zeros((strips, increments, repetitions, time))

        self.polarisation_data = self.initialize_data(strips, increments, repetitions, time)
        self.rotation_data = self.initialize_data(strips, increments, repetitions, time)
        self.average_polarisations = np.zeros((strips, increments))
        self.average_rotations = np.zeros((strips, increments))
        self.polarisation_errors = np.zeros((strips, increments))
        self.rotation_errors = np.zeros((strips, increments))
    
    def initialize_data(self, strips, increments, repetitions, time):
        return np.array([[[[np.zeros(time), np.zeros(8)] for _ in range(repetitions)] for _ in range(increments)] for _ in range(strips)], dtype=object)

    def update_parameters(self, strip, increment, repetition, parameters):
        self.polarisation_data[strip][increment][repetition][1] = parameters 
        self.rotation_data[strip][increment][repetition][1] = parameters

    def update_data(self, population, predator, strip, increment, repetition, time_step):
        self.position_data[strip, increment, repetition, time_step, :] = population.population_positions[:-1]
        self.direction_data[strip, increment, repetition, time_step, :] = population.population_directions[:-1]
        self.density_data[strip, increment, repetition, time_step, :] = population.population_densities

        self.predator_position_data[strip, increment, repetition, time_step, :] = predator.position
        self.predator_direction_data[strip, increment, repetition, time_step, :] = predator.direction
        self.predator_prey_distances[strip, increment, repetition, time_step] = predator.minimum_distance_to_prey
        self.predator_attack_number[strip, increment, repetition, time_step] = predator.attack_number

        self.polarisation_data[strip][increment][repetition][0][time_step] = population.polarisation
        self.rotation_data[strip][increment][repetition][0][time_step] = population.rotation

    def calculate_averages(self, strip, increment, samples):
        polarisation_samples = [repetition[-samples:] for repetition in self.get_polarisation_data()[strip][increment][:,0]]
        self.average_polarisations[strip][increment] = np.mean(polarisation_samples)

        rotation_samples = [repetition[-samples:] for repetition in self.get_rotation_data()[strip][increment][:,0]]
        self.average_rotations[strip][increment] = np.mean(rotation_samples)

    def calculate_errors(self, strip, increment, repetitions, samples):
        rotation_samples = [repetition[-samples:] for repetition in self.get_rotation_data()[strip][increment][:,0]]
        polarisation_samples = [repetition[-samples:] for repetition in self.get_polarisation_data()[strip][increment][:,0]]
        Rvars = np.var(rotation_samples, ddof=1, axis=1)
        Pvars = np.var(polarisation_samples, ddof=1, axis=1)

        Rpooled_stdev = np.sqrt(np.sum(Rvars)/repetitions)
        Ppooled_stdev = np.sqrt(np.sum(Pvars)/repetitions)

        Rpooled_se = Rpooled_stdev*np.sqrt(repetitions/samples)
        Ppooled_se = Ppooled_stdev*np.sqrt(repetitions/samples)

        self.rotation_errors[strip][increment] = Rpooled_se
        self.polarisation_errors[strip][increment] = Ppooled_se

    def get_polarisation_data(self):
        return self.polarisation_data
    
    def get_rotation_data(self):
        return self.rotation_data
    
    def get_position_data(self):
        return self.position_data
    
    def get_direction_data(self):
        return self.direction_data
    
    def get_polarisation_averages(self):
        return self.average_polarisations
    
    def get_rotation_averages(self):
        return self.average_rotations
    
    def get_rotation_errors(self):
        return self.rotation_errors
    
    def get_polarisation_errors(self):
        return self.polarisation_errors
    
    def get_predator_positions(self):
        return self.predator_position_data
    
    def get_predator_directions(self):
        return self.predator_direction_data
    
    def get_predator_prey_distances(self):
        return self.predator_prey_distances
    
    def get_predator_attack_number(self):
        return self.predator_attack_number
    
    def get_density_data(self):
        return self.density_data
    

#data = DataRecorder(5, 3, 10, 4, 2, 3)

'''print(f"All: \n{data.get_polarisation_data()}")
print(f"First strip: \n{data.get_polarisation_data()[0]}")
print(f"First increment: \n{data.get_polarisation_data()[0][0]}")
print(f"All repetitions in first increment: \n{data.get_polarisation_data()[0][0][:,0]}")
L4 = [repetition[-4:] for repetition in data.get_polarisation_data()[0][0][:,0]]
print(f"Last 4 elements in each repetition array: \n{L4}")
print(f"Repetition averages: \n{np.mean(L4)}")'''