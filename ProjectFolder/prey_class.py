import numpy as np
import random
from parent_class import Parent
from environment_class import Environment
from predator_class import Predator

'''This script contains the Prey class, which inherits from the Parent class, and is used to create instances of prey agents.
Differs from the Predator class in that it has a method to steer away from predators, and update its directionection vector accordingly.'''

class Prey(Parent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.predator_separation_vector = np.zeros(2)

    def calculate_predator_separation_vector(self, agents):
        number_of_neighbours = 0
        for agent in agents:
            if isinstance(agent, Predator) and self.calculate_distance_to_agents(agent) <= self.neighbourhood:
                self.predator_separation_vector += (self.position - agent.position)/(self.calculate_distance_to_agents(agent))
                number_of_neighbours += 1
        if number_of_neighbours > 0:
            self.predator_separation_vector /= number_of_neighbours
        return self.predator_separation_vector





