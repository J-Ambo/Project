import numpy as np
from parent_class import Parent
from parent_class import Parent
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder
import time
from line_profiler import profile
'''This script contains the main sinmulation loop, saving data for later plotting and analysis.'''

POPULATION = 100
NEIGHBOURS = 50    #number of neighbours each agent is influenced by
ARENA_RADIUS = 80
TIMESTEPS = 500
DIMENSIONS = 3
REPETITIONS = 3
number_of_increments = 2

env = Environment(ARENA_RADIUS, DIMENSIONS)
data_recorder = DataRecorder(POPULATION, DIMENSIONS, TIMESTEPS, REPETITIONS, number_of_increments)

def run_model():
    for i in range(number_of_increments):
        for r in range(REPETITIONS):
            parameter_array = np.array([POPULATION, NEIGHBOURS, ARENA_RADIUS, TIMESTEPS, DIMENSIONS, REPETITIONS, Parent.ral, Parent.rat])
            pop = Population(population_size=POPULATION, number_of_neighbours=NEIGHBOURS, environment=env)
            data_recorder.update_parameters(i, r, parameter_array)
            print(f"Increment {i+1}")
            print(f"Repetition {r+1}")
            
            for t in range(TIMESTEPS):
                pop.update_positions(env)
                data_recorder.update_data(pop, i, r, t)

        Parent.increment_rat(1)
        Parent.increment_ral(1)  #increment the radius of alignment
        
start_time = time.time()
run_model()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

Pol_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PolarisationData"
R_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\RotationData"
Pos_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PositionData"
np.save(f'{Pol_path}/polarisation_data_2', data_recorder.get_data()[1])
np.save(f'{R_path}/rotation_data_2', data_recorder.get_data()[2])
np.save(f'{Pos_path}/position_data_2', data_recorder.get_data()[0][:,:,:,:,0])

