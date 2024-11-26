from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment
from population_class import Population
from data_class import DataRecorder

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 100
ARENA_RADIUS = 50
TIMESTEPS = 500

env = Environment(ARENA_RADIUS)
pop = Population(population_size=POPULATION, number_of_neighbours=60, environment=env)
all_positions = pop.population_positions
data = DataRecorder(pop, TIMESTEPS)

#Animation using plt.pause method
fig, ax1 = plt.subplots(figsize=(5, 5))
ax1.set_axis_off()
ax1.set_xlim(-env.radius*1.01, env.radius*1.01)
ax1.set_ylim(-env.radius*1.01, env.radius*1.01)

centre = [0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax1.plot(x, y, c='black')

scatt = ax1.scatter([pos[0] for pos in all_positions],
            [pos[1] for pos in all_positions],
            c=['blue' if isinstance(agent, Prey) else 'red' for agent in pop.population_array],
           s=10)

for t in range(TIMESTEPS):            #Update the scatter plot for each timestep
    pop.update_positions(env)
    data.update_data(time=t)
    scatt.set_offsets([(pos[0], pos[1]) for pos in all_positions])
    plt.pause(0.001)

polarisation_data = data.get_data()[1]
rotation_data = data.get_data()[2]

fig, ax = plt.subplots()
time = np.linspace(0, TIMESTEPS, TIMESTEPS)
ax.set_ylim(-0.05,1.05)
ax.plot(time, polarisation_data, label='Polarisation', c='red')
ax.plot(time, rotation_data, label='Rotation', c='blue')
ax.legend()

plt.show()