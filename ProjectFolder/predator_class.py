import numpy as np
from parent_class import Parent
from environment_class import Environment
from population_class import Population
'''This script contains the Predator class, which inherits from the Parent class, and is used to create instances of predator agents.
Differs from the Prey class in that it steers towards prey.'''

'''NOTE: Boundary conditions and steering conditions are not yet up to date with the Prey class.'''

class Predator(Parent):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        self.direction = np.array([0,0,1])
        self.speed = 2*self.speed
        self.minimum_distance_to_prey = None   #np.array([])
        self.fixed_direction = None
        self.skip = False
        self.infront = True

    def find_neighbours(self, tree, population):
        neighbours, distances = tree.query_radius([self.position], 20, return_distance=True)
        neighbours = neighbours[distances!=0]
        distances = distances[distances!=0]
        if neighbours[distances!=0].size == 0:
            print('early exit')
            return np.array([]), np.array([])
        
        directions_to_neighbours = (neighbours - self.position)
        np.divide(directions_to_neighbours, distances[:,np.newaxis], out=directions_to_neighbours)

        dot_products = np.dot(directions_to_neighbours, neighbours)
        dot_products = np.round(dot_products, 3)

        angles_to_neighbours = np.arccos(dot_products)
       # print(angles_to_neighbours)

        mask = angles_to_neighbours <= self.half_perception_angle
        visible_indices = neighbours[mask]
        visible_distances = distances[mask]
        return neighbours[distances!=0], distances

    def calculate_direction(self, tree, population):
        vector_to_com = population.average_school_position - self.position
        distance_to_com = np.linalg.norm(vector_to_com) 
        
        if distance_to_com > 5 and self.skip == False:  
            self.direction = vector_to_com / distance_to_com
            #print('distance to COM ',distance_to_com, self.direction)

        elif distance_to_com <= 5 or self.skip == True:
            self.skip = True
            self.fixed_direction = self.direction
   
    def fnc(self, tree, population):
        neighbours, distances = self.find_neighbours(tree, population)
       # print(neighbours)

        distances_to_prey = np.linalg.norm(population.inlier_positions - self.position, axis=1)
        predator_school_vectors = (population.inlier_positions - self.position) / distances_to_prey[:,np.newaxis]
        dot_products = np.dot(predator_school_vectors, self.direction)
       
        if np.any(dot_products < 0) and (self.infront == True) and not np.all(dot_products < 0):
            self.minimum_distance_to_prey = min(distances_to_prey)

        elif np.all(dot_products < 0) or self.infront == False:
            self.infront = False
            self.minimum_distance_to_prey = None

    def update_predator(self, tree, population):
        self.fnc(tree, population)
        self.calculate_direction(tree, population)
        if self.fixed_direction is not None:
            self.position += self.fixed_direction * self.speed
        else:
            self.position += self.direction * self.speed
