import numpy as np
import random


class Environment:
    def _init_(self, size):
        self.size = size
    
    def create_environment(self):
        self.environment = np.zeros(self.size, self.size)
        return self.environment


class Human:
    def _init_(self, x, y, opinion):
        self.pos = np.array([x,y])
        self.opinion = opinion
        #self.identity = 0

    def move(self, x, y):
        self.newx = Environment.size - (x + random.randint(-1, 1))%Environment.size
        self.newy = Environment.size - (y + random.randint(-1, 1))%Environment.size
        self.pos = np.array([self.newx, self.newy])
        return self.pos
    



