from matplotlib import pyplot as plt
import random
import numpy as np
from parent_class import Parent
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder
import time

'''This script contains the main sinmulation loop, saving data for later plotting and analysis.'''

POPULATION = 300
NEIGHBOURS = 200    #number of neighbours each agent is influenced by
ARENA_RADIUS = 150
TIMESTEPS = 10000
DIMENSIONS = 3
REPETITIONS = 5

env = Environment(ARENA_RADIUS, DIMENSIONS)
data_recorder = DataRecorder(POPULATION, DIMENSIONS, TIMESTEPS, REPETITIONS)

start_time = time.time()
for n in range(REPETITIONS):
    #Parent.increment_ral(n*1.5)     #increment the radius of alignment
    parameter_array = np.array([POPULATION, NEIGHBOURS, ARENA_RADIUS, TIMESTEPS, DIMENSIONS, REPETITIONS, Parent.ral])
    pop = Population(population_size=POPULATION, number_of_neighbours=NEIGHBOURS, environment=env)
    all_positions = pop.population_positions
    data_recorder.update_parameters(n, parameter_array)
    print(f"Repetition {n+1}")
    #print(f"ral {pop.population_array[0].get_ral()}")
    #print(f"self.ral {pop.population_array[0].radius_of_alignment}")
    
    for t in range(TIMESTEPS):
        pop.update_positions(env)
        data_recorder.update_data(pop, time=t, repetitions=n)

Pol_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PolarisationData"
R_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\RotationData"
Pos_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PositionData"
np.save(f'{Pol_path}/polarisation_data_', data_recorder.get_data()[1])
np.save(f'{R_path}/rotation_data_', data_recorder.get_data()[2])
np.save(f'{Pos_path}/position_data_', data_recorder.get_data()[0][:,:,:,0])

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")