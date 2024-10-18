import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Human_agent import Human
from Environment_agent import Environment
from model import Model

Human.opinion_threshold = 0.8

model_instance = Model(100, 50)

fig, ax = plt.subplots()
ax.set_axis_off()
ax.set_xlim(0, model_instance.size)
ax.set_ylim(0, model_instance.size)

scatter = ax.scatter([human.pos[0] for human in model_instance.humans], [human.pos[1] for human in model_instance.humans], c=[human.opinion for human in model_instance.humans], cmap='rainbow',s=[10*np.exp(human.scepticism*2) for human in model_instance.humans])

def update_plot_frame(frame):
    '''Each frame all humans move and a random pair interact'''
    model_instance.update_timestep()

    scatter.set_offsets([(human.pos[0], human.pos[1]) for human in model_instance.humans])
    scatter.set_array([human.opinion for human in model_instance.humans])

    return scatter

ani = animation.FuncAnimation(fig, update_plot_frame, frames=1000, interval=200, repeat=True)
plt.show(block=True)

from IPython.display import HTML
HTML(ani.to_jshtml())