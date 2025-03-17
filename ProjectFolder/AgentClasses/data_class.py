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
        self.predator_target_densities = np.zeros((strips, increments, repetitions, time, population_size))
        self.predator_success = np.zeros((strips, increments, repetitions, time))

        self.polarisation_data = np.zeros((strips, increments, repetitions, time))  #self.initialize_data(strips, increments, repetitions, time)
        self.rotation_data = np.zeros((strips, increments, repetitions, time))  #self.initialize_data(strips, increments, repetitions, time)
        self.polarisation_averages = np.zeros((strips, increments))
        self.rotation_averages = np.zeros((strips, increments))
        self.polarisation_errors = np.zeros((strips, increments))
        self.rotation_errors = np.zeros((strips, increments))
    
   # def initialize_data(self, strips, increments, repetitions, time):
    #    return np.array([[[[np.zeros(time), np.zeros(8)] for _ in range(repetitions)] for _ in range(increments)] for _ in range(strips)], dtype=object)

   # def update_parameters(self, strip, increment, repetition, parameters):
    #    self.polarisation_data[strip][increment][repetition][1] = parameters 
     #   self.rotation_data[strip][increment][repetition][1] = parameters

    def update_data(self, population, predator, strip, increment, repetition, time_step):
        self.position_data[strip, increment, repetition, time_step, :] = population.population_positions[:-1]
        self.direction_data[strip, increment, repetition, time_step, :] = population.population_directions[:-1]
        self.density_data[strip, increment, repetition, time_step, :] = population.population_densities
        self.polarisation_data[strip, increment, repetition, time_step] = population.polarisation
        self.rotation_data[strip, increment, repetition, time_step] = population.rotation
        
        self.predator_position_data[strip, increment, repetition, time_step, :] = predator.position
        self.predator_direction_data[strip, increment, repetition, time_step, :] = predator.direction
        self.predator_prey_distances[strip, increment, repetition, time_step] = predator.minimum_distance_to_prey
        self.predator_attack_number[strip, increment, repetition, time_step] = predator.attack_number
        self.predator_target_densities[strip, increment, repetition, time_step, :] = predator.neighbour_densities
        self.predator_success[strip, increment, repetition, time_step] = predator.success

    def calculate_averages(self, strip, increment, samples):
        polarisation_samples = [repetition[-samples:] for repetition in self.polarisation_data[strip][increment]]
        self.polarisation_averages[strip][increment] = np.mean(polarisation_samples)

        rotation_samples = [repetition[-samples:] for repetition in self.rotation_data[strip][increment]]
        self.rotation_averages[strip][increment] = np.mean(rotation_samples)

    def calculate_errors(self, strip, increment, repetitions, samples):
        rotation_samples = [repetition[-samples:] for repetition in self.rotation_data[strip][increment]]
        polarisation_samples = [repetition[-samples:] for repetition in self.polarisation_data[strip][increment]]
        Rvars = np.var(rotation_samples, ddof=1, axis=1)
        Pvars = np.var(polarisation_samples, ddof=1, axis=1)

        Rpooled_stdev = np.sqrt(np.sum(Rvars)/repetitions)
        Ppooled_stdev = np.sqrt(np.sum(Pvars)/repetitions)

        Rpooled_se = Rpooled_stdev*np.sqrt(repetitions/samples)
        Ppooled_se = Ppooled_stdev*np.sqrt(repetitions/samples)

        self.rotation_errors[strip][increment] = Rpooled_se
        self.polarisation_errors[strip][increment] = Ppooled_se  

#data = DataRecorder(5, 3, 10, 4, 2, 3)

'''print(f"All: \n{data.get_polarisation_data()}")
print(f"First strip: \n{data.get_polarisation_data()[0]}")
print(f"First increment: \n{data.get_polarisation_data()[0][0]}")
print(f"All repetitions in first increment: \n{data.get_polarisation_data()[0][0][:,0]}")
L4 = [repetition[-4:] for repetition in data.get_polarisation_data()[0][0][:,0]]
print(f"Last 4 elements in each repetition array: \n{L4}")
print(f"Repetition averages: \n{np.mean(L4)}")'''