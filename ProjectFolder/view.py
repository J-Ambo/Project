from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 10
ARENA_RADIUS = 30
TIMESTEPS = 100
DIMENSIONS = 3
REPETITIONS = 1

env = Environment(ARENA_RADIUS, DIMENSIONS)


#Animation using plt.pause method
fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
#ax1.set_axis_off()
ax1.set_xlim3d(-env.radius*1.01, env.radius*1.01)
ax1.set_ylim3d(-env.radius*1.01, env.radius*1.01)
ax1.set_zlim3d(-env.radius*1.01, env.radius*1.01)

centre = [0,0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
#ax1.plot(x, y, c='black')

'''
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
                        zdir='x', s=10, c='gray', alpha=0.4)'''

for n in range(REPETITIONS):
    pop = Population(population_size=POPULATION, number_of_neighbours=9, environment=env)
    all_positions = pop.population_positions
    data = DataRecorder(pop, TIMESTEPS, REPETITIONS)
    for t in range(TIMESTEPS):            #Update the scatter plot for each timestep
        pop.update_positions(env)
        data.update_data(time=t, repetitions=n)
        '''
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
        '''
        plt.pause(0.001)



polarisation_data = data.get_data()[1]
rotation_data = data.get_data()[2]

for n in range(REPETITIONS):
    fig, ax = plt.subplots()
    time = np.linspace(0, TIMESTEPS, TIMESTEPS)
    ax.set_ylim(-0.05,1.05)
    ax.plot(time, polarisation_data[n], label='Polarisation', c='red')
    ax.plot(time, rotation_data[n], label='Rotation', c='blue')
    ax.legend()

plt.show()