import numpy as np
from parent_class import Parent
from parent_class import Parent
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder
from predator_class import Predator
import time
import os
from line_profiler import profile
'''This script contains the main sinmulation loop, saving data for later plotting and analysis.'''

population = 150
arena_radius = 200
timesteps = 300
samples = 2   #number of timesteps to include in the averages(counted from the last timestep)
dimensions = 3
repetitions = 1
increments = 1
strips = 1
increment_size = 0.5

starting_ral = 4
starting_rat = 13
starting_speed = 3
Parent.ral = starting_ral
Parent.rat = starting_rat
Population.steering_error = 0.15
Parent.perception_angle = np.deg2rad(270)
Parent.maximal_turning_angle = np.deg2rad(40)
Parent.speed = starting_speed

save_data = True
predator_on = True

env = Environment(arena_radius, dimensions)
data_recorder = DataRecorder(population, dimensions, timesteps, repetitions, increments, strips)

def run_model():
    for s in range(strips):
        print(f"Strip {s+1}, Ral is {Parent.ral}, Rat is {Parent.rat}")
        for i in range(increments):
            for r in range(repetitions):
                parameter_array = np.array([population, arena_radius, timesteps, repetitions, increments, strips, Parent.ral, Parent.rat])
                predator = Predator(0, 0, -200, 3, predator_on)
                pop = Population(population, env, predator)
                data_recorder.update_parameters(s, i, r, parameter_array)
                
                for t in range(timesteps):
                    tree = pop.get_tree()
                    pop.update_positions(tree, env, predator)
                    predator.update_predator(tree, pop)
                    pop.update_all_positions(predator)
                    #print(pop.all_positions)
                    #print(pop.population_positions)
                   
                    pop.calculate_order_parameters()
                    data_recorder.update_data(pop, predator, s, i, r, t)

            data_recorder.calculate_averages(s, i, samples)
            data_recorder.calculate_errors(s, i, repetitions, samples)
            Parent.increment_rat(increment_size)  #increment the radius of attraction
            Parent.increment_ral(increment_size)  #increment the radius of alignment

        Parent.rat = starting_rat
        #Parent.increment_rat(increment_size*(n+1))
        Parent.ral = starting_ral
        Parent.speed += 0.4
        #Population.steering_error += 0.05
        #Parent.perception_angle -= np.deg2rad(30)
        #Parent.maximal_turning_angle += np.deg2rad(10)

start_time = time.time()
run_model()
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

finishing_ral = np.round(data_recorder.get_polarisation_data()[-1][-1][-1][1][-2], 1)
finishing_rat = np.round(data_recorder.get_polarisation_data()[-1][-1][-1][1][-1], 1)
finishing_speed = Parent.speed #- 0.5

## Save data
if save_data:
    time_dmHM = time.strftime('%d%m_%H%M')
    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Data/{time_dm}/{time_dmHM}'
    os.makedirs(new_folder_path, exist_ok=True)

    with open(f'{new_folder_path}/parameters.txt', 'w') as file:
        file.write(f'Population: {population}\n')
        file.write(f'Arena Radius: {arena_radius}\n')
        file.write(f'Timesteps: {timesteps}\n')
        file.write(f'Samples: {samples}\n')
        file.write(f'Dimensions: {dimensions}\n')
        file.write(f'Repetitions: {repetitions}\n')
        file.write(f'Increments: {increments}\n')
        file.write(f'Strips: {strips}\n')
        file.write('\n')
        file.write('The following are agent specific attributes:\n')
        file.write(f'Radius of repulsion: {Parent.rr}\n')
        file.write(f'Range of radius of alignment: {starting_ral}-{finishing_ral}\n')
        file.write(f'(Range of alignment zone widths: {starting_ral-Parent.rr}-{finishing_ral-Parent.rr})\n')
        file.write(f'Range of radius of attraction: {starting_rat}-{finishing_rat}\n')
        file.write(f'(Range of attraction zone widths: {starting_rat-starting_ral}-{finishing_rat-finishing_ral})\n')
        file.write(f'Speed: {starting_speed}-{finishing_speed}\n')
        file.write(f'Steering error: {Population.steering_error}\n')
        perception_angle = np.rad2deg(Parent.perception_angle)
        file.write(f'Perception angle: {perception_angle}deg\n')
        maximum_turning_angle = np.rad2deg(Parent.maximal_turning_angle) 
        file.write(f'Maximal turning angle: {maximum_turning_angle}deg/timestep') #({maximum_turning_angle/0.1}deg/s)\n')

    np.save(f'{new_folder_path}/polarisation_data', data_recorder.get_polarisation_data())
    np.save(f'{new_folder_path}/rotation_data', data_recorder.get_rotation_data())
    np.save(f'{new_folder_path}/position_data', data_recorder.get_position_data())
    np.save(f'{new_folder_path}/direction_data', data_recorder.get_direction_data())
    np.save(f'{new_folder_path}/polarisation_averages', data_recorder.get_polarisation_averages())
    np.save(f'{new_folder_path}/rotation_averages', data_recorder.get_rotation_averages())
    np.save(f'{new_folder_path}/rotation_errors', data_recorder.get_rotation_errors())
    np.save(f'{new_folder_path}/polarisation_errors', data_recorder.get_polarisation_errors())
    np.save(f'{new_folder_path}/predator_positions', data_recorder.get_predator_positions())
    np.save(f'{new_folder_path}/predator_directions', data_recorder.get_predator_directions())

