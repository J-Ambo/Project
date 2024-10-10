import numpy as np
import random


class Environment:
    def __init__(self, size):
        self.size = size
    
    def create_environment(self):
        self.environment = np.zeros((self.size, self.size))
        return self.environment

class Human:
    def __init__(self, x, y, opinion):
        self.pos = np.array([x,y])
        self.opinion = opinion
        #self.identity = 0

    def move(self, x, y, env_instance):
  
        self.newx = adjust_coordinate(x, env_instance.size)
        self.newy = adjust_coordinate(y, env_instance.size)

        self.pos = np.array([self.newx, self.newy])
        return self.pos

def adjust_coordinate(coord, size):
    '''Ensures agents do not leave the environment'''
    if coord >= size:
        return size - 1
    elif coord <= 0:
        return coord + 1
    else:
        return coord + random.randint(-1, 1)


