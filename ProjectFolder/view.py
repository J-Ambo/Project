from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 50
NEIGHBOURS = 20
ARENA_RADIUS = 40
TIMESTEPS = 500
DIMENSIONS = 3
REPETITIONS = 10


graph_on = False

env = Environment(ARENA_RADIUS, DIMENSIONS)
data_recorder = DataRecorder(POPULATION, DIMENSIONS, TIMESTEPS, REPETITIONS)


if graph_on:
    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    #ax1.set_axis_off()
    ax1.set_xlim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_ylim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_zlim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')


    scatter3D = ax1.scatter(np.array([pos[0] for pos in all_positions]),
                            np.array([pos[1] for pos in all_positions]), 
                            np.array([pos[2] for pos in all_positions]),
                            s=10,
                            c=['blue' if isinstance(agent, Prey) else 'red' for agent in pop.population_array])
    xyscatter = ax1.scatter(np.array([pos[0] for pos in all_positions]),
                            np.array([pos[1] for pos in all_positions]), 
                            np.full(pop.population_size,-env.radius), 
                            zdir='z', s=10, c='gray', alpha=0.4)
    xzscatter = ax1.scatter(np.array([pos[0]for pos in all_positions]), 
                            np.full(pop.population_size,env.radius), 
                            np.array([pos[2] for pos in all_positions]), 
                            zdir='y', s=10, c='gray', alpha=0.4)
    yzscatter = ax1.scatter(np.full(pop.population_size,-env.radius), 
                            np.array([pos[1] for pos in all_positions]), 
                            np.array([pos[2] for pos in all_positions]), 
                            zdir='x', s=10, c='gray', alpha=0.4)

for n in range(REPETITIONS):
    pop = Population(population_size=POPULATION, number_of_neighbours=NEIGHBOURS, environment=env)
    all_positions = pop.population_positions
    
    for t in range(TIMESTEPS):            #Update the scatter plot for each timestep
        pop.update_positions(env)
        data_recorder.update_data(pop, time=t, repetitions=n)
        print(f"Time: {t}, Repetition = {n}")
        print(data_recorder.get_data()[1])
        
        if graph_on:
            scatter3D._offsets3d = (np.array([pos[0] for pos in pop.population_positions]),
                                    np.array([pos[1] for pos in pop.population_positions]), 
                                    np.array([pos[2] for pos in pop.population_positions]))
            xyscatter._offsets3d = (np.array([pos[0] for pos in pop.population_positions]), 
                                    np.array([pos[1] for pos in pop.population_positions]),
                                    np.full(pop.population_size,-env.radius))
            xzscatter._offsets3d = (np.array([pos[0] for pos in pop.population_positions]),
                                    np.full(pop.population_size, env.radius),
                                    np.array([pos[2] for pos in pop.population_positions]))
            yzscatter._offsets3d = (np.full(pop.population_size,-env.radius),
                                    np.array([pos[1] for pos in pop.population_positions]),
                                    np.array([pos[2] for pos in pop.population_positions]))
            ax1.set_title(f'Time: {t}')
            
            plt.pause(0.001)


polarisation_data = data_recorder.get_data()[1]
rotation_data = data_recorder.get_data()[2]
print(f"Polarisation data: {polarisation_data}")

time = np.linspace(0, TIMESTEPS, TIMESTEPS)
for r in range(REPETITIONS):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.set_ylim(-0.05,1.05)
    ax.plot(time, polarisation_data[r], label='Polarisation', c='red')
    ax.plot(time, rotation_data[r], label='Rotation', c='blue')
    ax.legend()
    ax.set_title(f'Repetition {r+1}')

plt.show()