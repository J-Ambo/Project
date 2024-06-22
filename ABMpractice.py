import matplotlib
matplotlib.use('TkAgg')
from pylab import *


n = 1000 # number of agents
r = 0.1 # neighborhood radius   
th = 0.5 # threshold for moving

class agent:
    pass

def initialize():
    global agents
    agents = []
    for i in range(n):
        ag = agent()
        ag.type = randint(2)
        ag.x = random()
        ag.y = random()
        agents.append(ag)

def observe():
    global agents
    cla()
    red = [ag for ag in agents if ag.type == 0]
    blue = [ag for ag in agents if ag.type == 1]

    plot([ag.x for ag in red], [ag.y for ag in red], 'ro')
    plot([ag.x for ag in blue], [ag.y for ag in blue], 'bo')
    axis('image')
    axis([0, 1, 0, 1])

def update():
    global agents
    ag = choice(agents)
    neighbours = [nb for nb in agents if (ag.x - nb.x)**2 + (ag.y - nb.y)**2 < r**2 and nb != ag]
    if len(neighbours) > 0:
        if len([nb for nb in neighbours if nb.type == ag.type]) / len(neighbours) < th: # if the the fraction of like neighbours is less than the threshold
            ag.x, ag.y = random(), random() # move to random position

import pycxsimulator
pycxsimulator.GUI().start(func=[initialize, observe, update])