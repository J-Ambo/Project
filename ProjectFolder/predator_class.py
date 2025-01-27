import numpy as np
from parent_class import Parent
from environment_class import Environment
'''This script contains the Predator class, which inherits from the Parent class, and is used to create instances of predator agents.
Differs from the Prey class in that it steers towards prey.'''

'''NOTE: Boundary conditions and steering conditions are not yet up to date with the Prey class.'''

class Predator(Parent):
    def __init__(self, x, y, z, dimensions, on):
        super().__init__(x, y, z, dimensions)
        self.position = self.position.astype(np.float64)
        self.speed = 0.9*self.speed * on
        
    def calculate_hunting_vector(self, population):
        average_prey_position = population.average_school_position
        #print(average_prey_position)
        hunting_vector = (average_prey_position - self.position) / np.linalg.norm(average_prey_position - self.position) + 1e-6
        return hunting_vector
        
    def update_predator(self, population):
        self.direction = self.calculate_hunting_vector(population)
        #print(self.direction)
        self.position += self.speed * self.direction
        #print(self.position)