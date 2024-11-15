import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from environment_class import Environment
from predator_class import Predator
from prey_class import Prey
'''This script is an alternative to the plt.pause method for creating an animation of the model.'''

POPULATION = 5
ARENA_RADIUS = 70
TIMESTEPS = 500

env = Environment(ARENA_RADIUS)
all_predators = []
for _ in range(0):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r * np.cos(theta)*0.9
    y = r * np.sin(theta)*0.9
    all_predators.append(Predator(x, y))

all_prey = []
for _ in range(POPULATION):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r * np.cos(theta)*0.9
    y = r * np.sin(theta)*0.9
    all_prey.append(Prey(x, y))

all_agents = all_prey + all_predators

#Animation using FuncAnimation method
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

#Update function for the animation
def update_frames(frame):
    all_steering_vectors = np.zeros((len(all_agents), 4, 2))
    
    for index, agent in enumerate(all_agents): 
        steering_vector = agent.calculate_steering_vector(all_agents, env)      
        all_steering_vectors[index] = steering_vector

    for index, agent in enumerate(all_agents):
        agent.update_position(all_steering_vectors[index])

    scatt.set_offsets([(agent.position[0], agent.position[1]) for agent in all_agents])
    return scatt

anim = animation.FuncAnimation(fig, update_frames, frames=TIMESTEPS, interval=50, repeat=True)
plt.show()

from IPython.display import HTML
HTML(anim.to_jshtml())
