import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from environment_class import Environment
from predator_class import Predator
from prey_class import Prey

#Create an environment instance
env = Environment(25)

#Create a list of birds
all_predators = []
for _ in range(0):
    x = random.uniform(-env.size*0.5, env.size*0.5)
    y = random.uniform(-env.size*0.5, env.size*0.5)
    all_predators.append(Predator(x, y))

all_prey = []
for _ in range(30):
    x = random.uniform(-env.size*0.5, env.size*0.5)
    y = random.uniform(-env.size*0.5, env.size*0.5)
    all_prey.append(Prey(x, y))

all_birds = all_prey + all_predators

#Animation using FuncAnimation method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.size*0.52, env.size*0.52)
ax.set_ylim(-env.size*0.52, env.size*0.52)
scatt = ax.scatter([bird.pos[0] for bird in all_birds],
                    [bird.pos[1] for bird in all_birds],
                    c=['blue' if isinstance(bird, Prey) else 'red' for bird in all_birds],
                      s=10)
centre = [0,0]
radius = env.size*0.5
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

    # Update the scatter plot data
    scatt.set_offsets([(bird.pos[0], bird.pos[1]) for bird in all_birds])
    return scatt

# Create the animation
anim = animation.FuncAnimation(fig, update_frames, frames=500, interval=50, repeat=True)
plt.show()
from IPython.display import HTML
HTML(anim.to_jshtml())
