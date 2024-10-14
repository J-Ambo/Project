import numpy as np
import random


class Environment:
    def __init__(self, size):
        self.size = size
    
    def create_environment(self):
        self.environment = np.zeros((self.size, self.size))
        return self.environment

class Human:
    opinion_threshold = 0.2

    def __init__(self, x, y, opinion, scepticism):
        self.pos = np.array([x,y])
        self.opinion = opinion
        self.scepticism = scepticism
        #self.influence

    def move(self, x, y, env_instance):
  
        self.newx = adjust_coordinate(x, env_instance.size)
        self.newy = adjust_coordinate(y, env_instance.size)

        self.pos = np.array([self.newx, self.newy])
        return self.pos

    def calculate_distance(self, other_human_instance):
        return ((self.pos[0] - other_human_instance.pos[0])**2 + (self.pos[1] - other_human_instance.pos[1])**2)**0.5

    def interact(self, other_human_instance):
        if abs(self.opinion - other_human_instance.opinion) <= Human.opinion_threshold:
            self.opinion += (1 - self.scepticism) * (other_human_instance.opinion - self.opinion)
            other_human_instance.opinion += (1 - other_human_instance.scepticism) * (self.opinion - other_human_instance.opinion)
        else:
            self.opinion += 0
            other_human_instance.opinion += 0


def adjust_coordinate(coord, size):
    '''Ensures agents do not leave the environment'''
    if coord >= size:
        return size - 1
    elif coord <= 0:
        return coord + 1
    else:
        return coord + random.randint(-1, 1)

