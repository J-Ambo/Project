import random
import numpy as np
from Human_agent import Human
from Environment_agent import Environment

class Model:
    def __init__(self, population, size, time):
        self.population = population
        self.size = size
        self.frames = time
        self.humans = np.array([])
        self.env = Environment(size)
        #self.seed = random.randint(0, 1000)
        self.create_humans()

    def create_humans(self):
        '''Function to create humans with random opinions and scepticism'''
        for i in range(self.population):
            x = random.randint(0, self.env.size)
            y = random.randint(0, self.env.size)
            op = random.uniform(0, 1)
            sc = 0 #random.uniform(0, 1)

            human = Human(x, y, op, sc)
            self.humans = np.append(self.humans, human)

    def update_timestep(self):
        '''Function to update the model each timestep'''
        #for human in self.humans:
            #human.move(human.pos[0], human.pos[1], self.env.size)

        random_index = random.randint(0, self.population)
        random_human = self.humans[random_index - 1]
        random_human.interact(random.choice(self.humans))
        #print(random_index)
        #print(self.log_data())

    def log_data(self):
        '''Function to log data for each human'''
        opinion_data = np.array([])
        for human in self.humans:
            opinion_data = np.append(opinion_data, human.opinion)
        return opinion_data
    






