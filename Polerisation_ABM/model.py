import numpy as np
import matplotlib.pyplot as plt
import random
from Polerisation_ABM.model_entities import Environment, Human


env = Environment(100)
env.create_environment()

humans = []
for i in range(100):
    x = random.randint(0,100)
    y = random.randint(0,100)
    op = random.randint(-1,1)
    human = Human(x, y, op)
    humans.append(human)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

scatter = ax.scatter([human.pos[0] for human in humans], [human.pos[1] for human in humans], c=['blue' if human.opinion == 1 else 'red' if human.opinion == -1 else 'green' for human in humans])

#for n in range (100):
    #for human in humans:
        #human.move(human.pos[0], human.pos[1])

    
plt.show()