from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause between each iteration to update the plot.'''

POPULATION = 5
ARENA_RADIUS = 70
TIMESTEPS = 500

env = Environment(ARENA_RADIUS)
all_agents = []

for _ in range(POPULATION):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r*np.cos(theta)*0.9
    y = r*np.sin(theta)*0.9
    all_agents.append(Prey(x, y))

#Animation using plt.pause method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.radius*1.01, env.radius*1.01)
ax.set_ylim(-env.radius*1.01, env.radius*1.01)

scatt = ax.scatter([agent.position[0] for agent in all_agents],
            [agent.position[1] for agent in all_agents],
            c=['blue' if isinstance(agent, Prey) else 'red' for agent in all_agents],
           s=10)

centre = [0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax.plot(x, y, c='black')

for _ in range(TIMESTEPS):            #Update the scatter plot for each timestep
    all_steering_vectors = np.zeros((len(all_agents), 4, 2))
    
    for index, agent in enumerate(all_agents): 
        steering_vector = agent.calculate_steering_vector(all_agents, env)      
        all_steering_vectors[index] = steering_vector

    for index, agent in enumerate(all_agents):
        agent.update_position(all_steering_vectors[index])

    scatt.set_offsets([(agent.position[0], agent.position[1]) for agent in all_agents])

    plt.pause(0.001)
plt.show()
