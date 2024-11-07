from matplotlib import pyplot as plt
import random
import numpy as np
from prey_class import Prey
from environment_class import Environment

'''This script is a alternative to matplotlib.animation.FuncAnimation for creating
 an animation of the model. It uses plt.pause method to update the scatter plot'''

env = Environment(50)
agents = []
for _ in range(20):
    x = random.uniform(-env.size*0.5, env.size*0.5)
    y = random.uniform(-env.size*0.5, env.size*0.5)
    agents.append(Prey(x, y))

#Animation using plt.pause method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.size*0.5, env.size*0.5)
ax.set_ylim(-env.size*0.5, env.size*0.5)
scatt = ax.scatter([agent.pos[0] for agent in agents],
                    [agent.pos[1] for agent in agents],
                    c=['blue' if isinstance(agent, Prey) else 'red' for agent in agents],
                      s=10)
centre = [0,0]
radius = env.size*0.5
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax.plot(x, y, c='black')

for _ in range(500):            #Update the scatter plot for each iteration
    for agent in agents:
        agent.update_prey(agents, env)
    
    scatt.set_offsets([(agent.pos[0], agent.pos[1]) for agent in agents])

    plt.pause(0.001)
plt.show()
