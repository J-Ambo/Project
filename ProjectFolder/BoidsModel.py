import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from environment_class import Environment
from predator_class import Predator
from prey_class import Prey
'''This script is an alternative to the plt.pause method for creating an animation of the model.'''

env = Environment(50)
all_predators = []
for _ in range(0):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    all_predators.append(Predator(x, y))

all_prey = []
for _ in range(1):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    all_prey.append(Prey(x, y))

all_birds = all_prey + all_predators

#Animation using FuncAnimation method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.radius*1.01, env.radius*1.01)
ax.set_ylim(-env.radius*1.01, env.radius*1.01)
scatt = ax.scatter([bird.pos[0] for bird in all_birds],
                    [bird.pos[1] for bird in all_birds],
                    c=['blue' if isinstance(bird, Prey) else 'red' for bird in all_birds],
                      s=10)
centre = [0,0]
radius = env.radius
theta = np.linspace(0, 2*np.pi, 100)
x = centre[0] + radius * np.cos(theta)
y = centre[1] + radius * np.sin(theta)
ax.plot(x, y, c='black')

#Update function for the animation
def update_frames(frame):
    for prey in all_prey:
        prey.update_prey(all_birds, env)

    for predator in all_predators:
        predator.update_predator(all_birds, env)

    scatt.set_offsets([(bird.pos[0], bird.pos[1]) for bird in all_birds])
    return scatt

anim = animation.FuncAnimation(fig, update_frames, frames=500, interval=50, repeat=True)
plt.show()

from IPython.display import HTML
HTML(anim.to_jshtml())