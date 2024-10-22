import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Human_agent import Human
from Environment_agent import Environment
from model import Model

Human.opinion_threshold = 1

model_instance = Model(5, 50, 10)
print(model_instance.log_data())
#model_instance.run_model(model_instance.frames)

fig0, ax0 = plt.subplots()

average_opinion_all_frames = np.array([])
for frame in range(model_instance.frames):
    model_instance.update_timestep()
    average_opinion = sum(model_instance.log_data())/model_instance.population
    average_opinion_all_frames = np.append(average_opinion_all_frames, average_opinion)

    #print(model_instance.log_data())
    
ax0.plot(range(model_instance.frames), average_opinion_all_frames)  



print(model_instance.log_data())


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

ani = animation.FuncAnimation(fig, update_plot_frame, frames=model_instance.frames, interval=200, repeat=True)
plt.show(block=True)

from IPython.display import HTML
HTML(ani.to_jshtml())