import numpy as np
import random

class Human:
    opinion_threshold = 0.2

    def __init__(self, x, y, opinion, scepticism):
        '''Human instance attributes upon creation'''
        self.pos = np.array([x,y])
        self.opinion = opinion
        self.scepticism = scepticism
        #self.influence

    def adjust_coordinate(self, coord, environment_instance_size):
        '''Ensures agents do not leave the environment'''
        if coord >= environment_instance_size:
            return environment_instance_size - 1
        elif coord <= 0:
            return coord + 1
        else:
            return coord + random.randint(-1, 1)

    def move(self, x, y, env_instance_size):
        '''Moves the human to a new position'''
        self.newx = self.adjust_coordinate(x, env_instance_size)
        self.newy = self.adjust_coordinate(y, env_instance_size)

        self.pos = np.array([self.newx, self.newy])
        return self.pos

    def interact(self, other_human_instance):
        '''Interaction between two humans, updating their opinions'''
        if abs(self.opinion - other_human_instance.opinion) <= Human.opinion_threshold:
            self.opinion += (1 - self.scepticism) * (other_human_instance.opinion - self.opinion)
            other_human_instance.opinion += (1 - other_human_instance.scepticism) * (self.opinion - other_human_instance.opinion)
        else:
            self.opinion += 0
            other_human_instance.opinion += 0

    def log_data(self):
        '''Function to log data for each human'''
        opinion_data = []
        opinion_data.append(self.opinion)
        return opinion_data


