from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment
from population_class import Population

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 100
ARENA_RADIUS = 55
TIMESTEPS = 2000

env = Environment(ARENA_RADIUS)
pop = Population(population_size=POPULATION, number_of_neighbours=20, environment=env)
all_positions = pop.population_positions

#Animation using plt.pause method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.radius*1.01, env.radius*1.01)
ax.set_ylim(-env.radius*1.01, env.radius*1.01)

centre = [0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax.plot(x, y, c='black')

scatt = ax.scatter([pos[0] for pos in all_positions],
            [pos[1] for pos in all_positions],
            c=['blue' if isinstance(agent, Prey) else 'red' for agent in pop.population_array],
           s=10)

for _ in range(TIMESTEPS):            #Update the scatter plot for each timestep
    pop.update_positions(env)
    scatt.set_offsets([(pos[0], pos[1]) for pos in all_positions])
    #print(f"Positions: {pop.population_positions}")
    plt.pause(0.001)
plt.show()
