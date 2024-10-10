import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import sys
#sys.path.append('C:/Users/44771/Documents/Level4Project/Polerisation_ABM/model.py')
from model_entities import Environment, Human
random.seed(0)


env = Environment(100)
env.create_environment()

humans = []
for i in range(100):
    x = random.randint(0,100)
    y = random.randint(0,100)
    op = random.choice([-1,1])
    human = Human(x, y, op)
    humans.append(human)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


scatter = ax.scatter([human.pos[0] for human in humans], [human.pos[1] for human in humans], c=[human.opinion for human in humans], cmap='bwr')

def update_frame(frame):
    for human in humans:
        human.move(human.pos[0], human.pos[1], env)
    scatter.set_offsets([(human.pos[0], human.pos[1]) for human in humans])
    return scatter
ani = animation.FuncAnimation(fig, update_frame, frames=100, interval=500, repeat=True)
plt.show(block=True)