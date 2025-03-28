import numpy as np
from AgentClasses.parent_class import Parent
from AgentClasses.environment_class import Environment
from AgentClasses.population_class import Population
from AgentClasses.data_class import DataRecorder
from AgentClasses.predator_class import Predator
import time
import os
from line_profiler import profile
from multiprocessing import Pool
import concurrent.futures
from numpy.random import PCG64, Generator
'''This script contains the main simulation loop, saving data for later plotting and analysis.'''

Population.population_size = 300
arena_radius = 2000
timesteps = 1000
samples = 2   #number of timesteps to include in the averages(counted from the last timestep)
dimensions = 3
depth = 10
repetitions = 1
increments = 1
strips = 1
increment_size = 0.4

starting_ral = 3 
starting_rat = 15
starting_speed = 1
Parent.rr = 1
Parent.ral = starting_ral
Parent.rat = starting_rat
Population.steering_error = 0.09
Parent.perception_angle = np.deg2rad(270)
Parent.maximal_turning_angle = np.deg2rad(60) * 0.1  #deg/s * s
Parent.evasion_angle = np.deg2rad(20)
#Parent.speed = starting_speed
Population.selfish = 0

#seeds = np.random.default_rng().integers(10000, 30000, size=3)

processes = 1
process_start = time.strftime('%d%m_%H%M')
start_dm = time.strftime('%d%m')
save_data = True
predator_on = False

env = Environment(arena_radius, dimensions, depth)
data_recorder = DataRecorder(Population.population_size, dimensions, timesteps, repetitions, increments, strips)
def run_simulation(args):
    Parent.speed, seed = args
    np.random.seed(seed)
    for s in range(strips):
        for i in range(increments):
            for r in range(repetitions):
                #print(r)
                predator = Predator(0, 0, -2005)
                if not predator_on:
                    predator.speed = 0

                pop = Population(Population.population_size, env, predator)
            
                for t in range(timesteps):
                    tree = pop.get_tree()
                    pop.update_positions(tree, env, predator)
                    pop.calculate_order_parameters()
                    
                    if predator_on:
                        predator.update_predator(tree, pop)

                    data_recorder.update_data(pop, predator, s, i, r, t)
                
                if save_data:
                    new_folder_path = f'C:/Users/44771/Desktop/Data/{start_dm}/{process_start}/Sp{np.round(Parent.speed, 1)}_{seed}'
                    os.makedirs(new_folder_path, exist_ok=True)

                    np.save(f'{new_folder_path}/polarisation_data', data_recorder.polarisation_data)
                    np.save(f'{new_folder_path}/rotation_data', data_recorder.rotation_data)
                    np.save(f'{new_folder_path}/position_data', data_recorder.position_data)
                    np.save(f'{new_folder_path}/direction_data', data_recorder.direction_data)
                   # np.save(f'{new_folder_path}/polarisation_averages', data_recorder.polarisation_averages)
                    #np.save(f'{new_folder_path}/rotation_averages', data_recorder.rotation_averages)
                   # np.save(f'{new_folder_path}/rotation_errors', data_recorder.rotation_errors)
                   # np.save(f'{new_folder_path}/polarisation_errors', data_recorder.polarisation_errors)
                   # np.save(f'{new_folder_path}/predator_positions', data_recorder.predator_position_data)
                    #np.save(f'{new_folder_path}/predator_directions', data_recorder.predator_direction_data)
                    #np.save(f'{new_folder_path}/predator_prey_distances', data_recorder.predator_prey_distances)
                   # np.save(f'{new_folder_path}/predator_attack_number', data_recorder.predator_attack_number)
                    np.save(f'{new_folder_path}/density_data', data_recorder.density_data)
                    #np.save(f'{new_folder_path}/target_density_data', data_recorder.predator_target_densities)
                    #np.save(f'{new_folder_path}/predator_success', data_recorder.predator_success)

                                    
                    with open(f'{new_folder_path}/parameters.txt', 'w') as file:
                        file.write(f'Population: {Population.population_size}\n')
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
                        file.write(f'Range of radius of alignment: {starting_ral}-{Parent.ral}\n')
                        file.write(f'(Range of alignment zone widths: {starting_ral-Parent.rr}-{Parent.ral-Parent.rr})\n')
                        file.write(f'Range of radius of attraction: {starting_rat}-{Parent.rat}\n')
                        file.write(f'(Range of attraction zone widths: {starting_rat-starting_ral}-{Parent.rat-Parent.ral})\n')
                        file.write(f'Speed: {starting_speed}-{Parent.speed}\n')
                        file.write(f'Steering error: {Population.steering_error}\n')
                        perception_angle = np.rad2deg(Parent.perception_angle)
                        file.write(f'Perception angle: {perception_angle}deg\n')
                        maximum_turning_angle = np.rad2deg(Parent.maximal_turning_angle) 
                        file.write(f'Maximal turning angle: {maximum_turning_angle}deg/timestep') #({maximum_turning_angle/0.1}deg/s)\n')
        
        data_recorder.calculate_averages(s, i, samples)
        data_recorder.calculate_errors(s, i, repetitions, samples)
    return f'Done, speed {Parent.speed}'

rng = Generator(PCG64())
integers = rng.integers(2**10, 2**32 -1, size=processes)
speeds = np.linspace(4, 5, processes)
s = time.perf_counter()
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        args = list(zip(speeds, integers))
        
        results = executor.map(run_simulation, args)

        for result in results:
            print(result)

f = time.perf_counter()
print(f'Finished in {round(f-s,2)}s')


#if __name__ == '__main__':
#    start_time = time.time()
#    with Pool() as pool:
#        args = [(r, repetitions, timesteps, strips) for r in range(repetitions)]
#        pool.map(run_simulation, args)
#    end_time = time.time()
#    execution_time = end_time - start_time
#    print(f"Execution time: {execution_time} seconds")



def run_model():
    for s in range(strips):
        print(f"Strip {s+1}, Ral is {Parent.ral}, Rat is {Parent.rat}")
        for i in range(increments):
            for r in range(repetitions):
               
                predator = Predator(0, 0, -200)
                if not predator_on:
                    predator.speed = 0

                pop = Population(Population.population_size, env, predator)
                
                for t in range(timesteps):
                    tree = pop.get_tree()
                    pop.update_positions(tree, env, predator)
                    pop.calculate_order_parameters()

                    predator.update_predator(tree, pop)
                    
                    data_recorder.update_data(pop, predator, s, i, r, t)
                
            data_recorder.calculate_averages(s, i, samples)
            data_recorder.calculate_errors(s, i, repetitions, samples)
           # Parent.increment_rat(increment_size)  #increment the radius of attraction
           # Parent.increment_ral(increment_size)  #increment the radius of alignment

        Parent.rat = starting_rat
        #Parent.increment_rat(increment_size*(n+1))
        Parent.ral = starting_ral
       # Parent.speed += 0.4
        #Population.steering_error += 0.05
        #Parent.perception_angle -= np.deg2rad(30)
        #Parent.maximal_turning_angle += np.deg2rad(10)
        #Parent.evasion_angle += np.deg2rad(5)
        Population.selfish += 1/strips


#start_time = time.time()
#run_model()
#end_time = time.time()
#execution_time = end_time - start_time
#print(f"Execution time: {execution_time} seconds")

#finishing_ral = Parent.ral 
#finishing_rat = Parent.rat
#finishing_speed = Parent.speed - 1.2


