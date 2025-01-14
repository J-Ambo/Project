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
arena_radius = 200
timesteps = 3000
samples = 1000
dimensions = 3
repetitions = 10
increments = 11
strips = 5
increment_size = 0.5

starting_ral = 1
starting_rat = 13
starting_speed = 3
Parent.ral = starting_ral
Parent.rat = starting_rat
Population.steering_error = 0.15
Parent.perception_angle = np.deg2rad(270)
Parent.maximal_turning_angle = np.deg2rad(40)
Parent.speed = starting_speed

save_data = True

env = Environment(arena_radius, dimensions)
data_recorder = DataRecorder(population, dimensions, timesteps, repetitions, increments, strips)

'''def run_model():
    for n in range(strips):
        print(f"Strip {n+1}, Ral is {Parent.ral}, Rat is {Parent.rat}")
        for i in range(increments):
            parameter_array = np.array([population, arena_radius, timesteps, repetitions, increments, strips, Parent.ral, Parent.rat])
            data_recorder.update_parameters(n, i, 0, parameter_array)
            print(f"Strip: {n+1}")
            print(f"Increment {i+1}")

            # Initialize populations for all repetitions
            populations = [Population(population, env) for _ in range(repetitions)]

            for t in range(timesteps):
                for r, pop in enumerate(populations):
                    pop.update_positions(env)
                    data_recorder.update_data(pop, n, i, r, t)

            data_recorder.update_averages(n, i, samples)
            Parent.increment_rat(increment_size)  # increment the radius of attraction
            Parent.increment_ral(increment_size)  # increment the radius of alignment

        Parent.rat = starting_rat
        Parent.increment_rat(increment_size * (n + 1))
        Parent.ral = starting_ral'''


def run_model():
    for n in range(strips):
        print(f"Strip {n+1}, Ral is {Parent.ral}, Rat is {Parent.rat}")
        print('Angle',Parent.maximal_turning_angle)
        for i in range(increments):
            for r in range(repetitions):
                parameter_array = np.array([population, arena_radius, timesteps, repetitions, increments, strips, Parent.ral, Parent.rat])
                pop = Population(population, env)
                data_recorder.update_parameters(n, i, r, parameter_array)
                print(f"Strip: {n+1}")
                print(f"Increment {i+1}")
                print(f"Repetition {r+1}")
                
                for t in range(timesteps):
                    tree = pop.get_tree()
                    neighbours_distances = pop.find_neighbours(tree)
                    neighbours = neighbours_distances[0]
                    distances = neighbours_distances[1]

                    pop.update_positions(env, neighbours, distances)
                    pop.calculate_order_parameters(tree, distances)
                    data_recorder.update_data(pop, n, i, r, t)

            data_recorder.calculate_averages(n, i, samples)
            data_recorder.calculate_errors(n, i, repetitions, samples)
            Parent.increment_rat(increment_size)  #increment the radius of attraction
            Parent.increment_ral(increment_size)  #increment the radius of alignment

        Parent.rat = starting_rat
        #Parent.increment_rat(increment_size*(n+1))
        Parent.ral = starting_ral
        Parent.speed += 0.5
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
finishing_speed = Parent.speed - 0.5

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
    np.save(f'{new_folder_path}/polarisation_averages', data_recorder.get_polarisation_averages())
    np.save(f'{new_folder_path}/rotation_averages', data_recorder.get_rotation_averages())
    np.save(f'{new_folder_path}/rotation_errors', data_recorder.get_rotation_errors())
    np.save(f'{new_folder_path}/polarisation_errors', data_recorder.get_polarisation_errors())


