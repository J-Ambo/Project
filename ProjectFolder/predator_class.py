import numpy as np
from parent_class import Parent
from environment_class import Environment
'''This script contains the Predator class, which inherits from the Parent class, and is used to create instances of predator agents.
Differs from the Prey class in that it steers towards prey.'''

'''NOTE: Boundary conditions and steering conditions are not yet up to date with the Prey class.'''

class Predator(Parent):
    def __init__(self, x, y, z, dimensions):
        super().__init__(x, y, z, dimensions)
        self.position = self.position.astype(np.float64)
        self.speed = 2*self.speed
        self.attack_radius = 20
        self.minimum_distance_to_prey = None   #np.array([])
        self.fixed_direction = None
        self.skip = False
        self.infront = True
        
    def calculate_direction(self, tree, population):
        vector_to_com = population.average_school_position - self.position
        distance_to_com = np.linalg.norm(vector_to_com) 

        if distance_to_com >= 1 and self.skip == False:  
            self.direction = vector_to_com / distance_to_com
            #print(False)

        elif distance_to_com <= 1 or self.skip == True:
            #print(True, distance_to_com)
            self.skip = True
            self.fixed_direction = self.direction
    
    def fnc(self, population):
        distances_to_prey = np.linalg.norm(population.inlier_positions - self.position, axis=1)
        predator_school_vectors = (population.inlier_positions - self.position) / distances_to_prey[:,np.newaxis]
        dot_products = np.dot(predator_school_vectors, self.direction)
        if (np.any(dot_products < 0) and not np.all(dot_products < 0)): #and (self.infront == True):
            self.minimum_distance_to_prey = min(distances_to_prey)

        elif np.all(dot_products < 0): #or self.infront == False:
            self.infront = False
            self.minimum_distance_to_prey = None

    def update_predator(self, tree, population):
        self.fnc(population)
        self.calculate_direction(tree, population)
        if self.fixed_direction is not None:
            self.position += self.fixed_direction * self.speed
        else:
            self.position += self.direction * self.speed
