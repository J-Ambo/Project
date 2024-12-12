import numpy as np
from parent_class import Parent
from parent_class import Parent
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder
import time
import os
from line_profiler import profile
'''This script contains the main sinmulation loop, saving data for later plotting and analysis.'''

population = 100
neighbours = 299   #number of neighbours each agent is influenced by
arena_radius = 200
timesteps = 200
dimensions = 3
repetitions = 1
increments = 1
increment_size = 0.5

save_data = True

env = Environment(arena_radius, dimensions)
data_recorder = DataRecorder(population, dimensions, timesteps, repetitions, increments)

def run_model():
    for i in range(increments):
        for r in range(repetitions):
            parameter_array = np.array([population, neighbours, arena_radius, timesteps, dimensions, repetitions, Parent.ral, Parent.rat])
            pop = Population(population, neighbours, env)
            data_recorder.update_parameters(i, r, parameter_array)
            print(f"Increment {i+1}")
            print(f"Repetition {r+1}")
            
            for t in range(timesteps):
                pop.update_positions(env)
                data_recorder.update_data(pop, i, r, t)

        Parent.increment_rat(increment_size)  #increment the radius of attraction
        Parent.increment_ral(increment_size)  #increment the radius of alignment

starting_ral = Parent.ral
starting_rat = Parent.rat

start_time = time.time()
run_model()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

finishing_ral = data_recorder.get_polarisation_data()[-1][-1][1][-2]
finishing_rat = data_recorder.get_polarisation_data()[-1][-1][1][-1]

## Save data
if save_data:
    time_dmHM = time.strftime('%d%m_%H%M')
    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Data/{time_dm}/{time_dmHM}'
    os.makedirs(new_folder_path, exist_ok=True)

    with open(f'{new_folder_path}/parameters.txt', 'w') as file:
        file.write(f'Population: {population}\n')
        file.write(f'Neighbours: {neighbours}\n')
        file.write(f'Arena Radius: {arena_radius}\n')
        file.write(f'Timesteps: {timesteps}\n')
        file.write(f'Dimensions: {dimensions}\n')
        file.write(f'Repetitions: {repetitions}\n')
        file.write(f'Increments: {increments}\n')
        file.write('\n')
        file.write('The following are agent specific attributes:\n')
        file.write(f'Radius of repulsion: {Parent.rr}\n')
        file.write(f'Radius of alignment range: {starting_ral}-{finishing_ral}\n')
        file.write(f'Radius of attraction range: {starting_rat}-{finishing_rat}\n')
        file.write(f'Speed: {Parent.speed}\n')
        perception_angle = np.rad2deg(Parent.perception_angle)
        file.write(f'Perception angle: {perception_angle}deg\n')
        maximum_turning_angle = np.rad2deg(Parent.maximal_turning_angle) 
        file.write(f'Maximal turning angle: {maximum_turning_angle}deg/timestep ({maximum_turning_angle/0.1}deg/s)\n')

    np.save(f'{new_folder_path}/polarisation_data', data_recorder.get_polarisation_data())
    np.save(f'{new_folder_path}/rotation_data', data_recorder.get_rotation_data())
    np.save(f'{new_folder_path}/position_data', data_recorder.get_position_data())

