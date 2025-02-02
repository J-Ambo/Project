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
        self.attack_radius = 20
    
    def calculate_hunting_vector(self, tree, population):     #calculates the vector towards the densest region
        max_density_position = self.calculate_densest_position(tree, population)
        hunting_vector = (max_density_position - self.position) / np.linalg.norm(max_density_position - self.position) + 1e-6
        return hunting_vector
    
    def calculate_attack_vector(self, tree, population):  # calculates the vector towards its nearest neighbour
        distances, indices = tree.query([self.position], k=2)
        nearest_distance = distances[0][1]
        nearest_index = indices[0][1]
        
        if nearest_distance <= self.attack_radius:
            nearest_position = population.population_positions[nearest_index]
            attack_vector = (nearest_position - self.position) / (nearest_distance + 1e-6)
            return attack_vector
        else:
            return np.zeros_like(self.position)

    def get_densities(self, tree, population):
        Nn = 20
        distances = tree.query(population.population_positions, k=Nn)[0]
        cache_key = hash(population.population_positions.tobytes())
        if cache_key in population._density_cache:
            return population._density_cache[cache_key]
        
        densities = (Nn-1)/distances[:, -1]
        population._density_cache[cache_key] = densities
        return densities
    
    def calculate_densest_position(self, tree, population):
        densities = self.get_densities(tree, population)
        return population.population_positions[np.argmax(densities)]
    
    def update_predator(self, tree, population):
        self.direction = self.calculate_hunting_vector(tree, population) + self.calculate_attack_vector(tree, population)
        #print(self.direction)
        self.position += self.speed * self.direction
        #print(self.position)