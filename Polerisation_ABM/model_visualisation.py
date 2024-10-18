import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Human_agent import Human
from Environment_agent import Environment

#random.seed(123567)
population = 5
Human.opinion_threshold = 0.8

env = Environment(100)
env.create_environment()

humans = []
for i in range(population):
    x = random.randint(0,env.size)
    y = random.randint(0,env.size)
    op = random.uniform(-1,1)   #random.choice([-1,1])
    sc = random.uniform(0,1) 

    human = Human(x, y, op, sc)
    humans.append(human)

fig, ax = plt.subplots()
ax.set_axis_off()
ax.set_xlim(0, env.size)
ax.set_ylim(0, env.size)

scatter = ax.scatter([human.pos[0] for human in humans], [human.pos[1] for human in humans], c=[human.opinion for human in humans], cmap='rainbow',s=[10*np.exp(human.scepticism*2) for human in humans])

def update_frame(frame):
    '''Each frame all humans move and a random pair interact'''
    for human in humans:
        human.move(human.pos[0], human.pos[1], env)
    
    random_index = random.randint(0,population)
    random_human = humans[random_index - 1]
    random_human.interact(random.choice(humans))

    scatter.set_offsets([(human.pos[0], human.pos[1]) for human in humans])
    scatter.set_array([human.opinion for human in humans])

    return scatter

ani = animation.FuncAnimation(fig, update_frame, frames=100, interval=100, repeat=True)
plt.show(block=True)

from IPython.display import HTML
HTML(ani.to_jshtml())