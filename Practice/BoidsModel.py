import random
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from environment_class import Environment
from predator_class import Predator
from prey_class import Prey

#Create an environment instance
env = Environment(100)

#Create a list of birds
all_predators = []
for _ in range(0):
    x = random.uniform(env.size*0.05, env.size*0.95)
    y = random.uniform(env.size*0.05, env.size*0.95)
    all_predators.append(Predator(x, y))

all_prey = []
for _ in range(2):
    x = random.uniform(env.size*0.05, env.size*0.95)
    y = random.uniform(env.size*0.05, env.size*0.95)
    all_prey.append(Prey(x, y))

all_birds = all_prey + all_predators

#Animation using FuncAnimation method
fig, ax = plt.subplots(figsize=(6,6))
ax.set_axis_off()
ax.set_xlim(-env.size*0.05, env.size*1.05)
ax.set_ylim(-env.size*0.05, env.size*1.05)
scatt = ax.scatter([bird.pos[0] for bird in all_birds],
                    [bird.pos[1] for bird in all_birds],
                    c=['blue' if isinstance(bird, Prey) else 'red' for bird in all_birds],
                      s=10)

ax.plot(np.zeros(100), np.linspace(0, env.size, 100), color='black')
ax.plot(np.full(100,env.size), np.linspace(0, env.size, 100), color='black')
ax.plot(np.linspace(0,env.size,100), np.zeros(100), color='black')
ax.plot(np.linspace(0, env.size, 100),np.full(100, env.size), color='black')

#Update function for the animation
def update_frames(frame):
    for prey in all_prey:
        #print(f"Position: {prey.pos}, Direction: {prey.dir}, steering vectors: {prey.calculate_steering_vector(all_birds)}")
        #print(prey.point_is_out_of_bounds(prey.pos[0], env))
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
