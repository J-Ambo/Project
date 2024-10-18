import random
from Human_agent import Human
from Environment_agent import Environment

class Model:
    def __init__(self, population, size):
        self.population = population
        self.size = size
        self.humans = []
        self.env = Environment(size)
        self.create_humans()

    def create_humans(self):
        '''Function to create humans with random opinions and scepticism'''
        for i in range(self.population):
            x = random.randint(0, self.env.size)
            y = random.randint(0, self.env.size)
            op = random.uniform(-1, 1)
            sc = random.uniform(0, 1)

            human = Human(x, y, op, sc)
            self.humans.append(human)

    def update_timestep(self):
        '''Function to update the model each timestep'''
        for human in self.humans:
            human.move(human.pos[0], human.pos[1], self.env.size)

        random_index = random.randint(0, self.population)
        random_human = self.humans[random_index - 1]
        random_human.interact(random.choice(self.humans))

    def log_data(self):
        '''Function to log data for each human'''
        opinion_data = []
        for human in self.humans:
            opinion_data.append(human.opinion)
        return opinion_data

    def run_model(self, frames):
        for frame in range(frames):
            self.update_frame()

        return self.log_data()
    






