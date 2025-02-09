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
        self.speed = 2*self.speed * on
        self.attack_radius = 20
        self.minimum_distance_to_prey = None   #np.array([])
        self.fixed_direction = None
    
    '''def calculate_hunting_vector(self, tree, population):     #calculates the vector towards the densest region
        #max_density_position = self.calculate_densest_position(tree, population)
        #hunting_vector = (max_density_position - self.position) / np.linalg.norm(max_density_position - self.position)
        hunting_vector = (population.average_school_position - self.position) / np.linalg.norm(population.average_school_position - self.position)
        return hunting_vector'''
    
    '''def calculate_attack_vector(self, tree, population):  # calculates the vector towards its nearest neighbour
        distances, indices = tree.query([self.position], k=2)
        nearest_distance = distances[0][1]
        nearest_index = indices[0][1]
        
        if nearest_distance <= self.attack_radius:
            nearest_position = population.population_positions[nearest_index]
            attack_vector = (nearest_position - self.position) / (nearest_distance)
            return attack_vector
        else:
            return np.zeros_like(self.position)'''
        
    def calculate_direction(self, tree, population):
        if not hasattr(self, 'skip'):
            self.skip = 0
        if self.skip == 0:  
            vector_to_com = population.average_school_position - self.position
            distance_to_com = np.linalg.norm(vector_to_com)  
            self.direction = vector_to_com / distance_to_com

        elif distance_to_com <= 20 and self.skip != 0:
            self.fixed_direction = self.direction
            self.skip = 1

        #else:
         #   self.skip += 1
          #  return self.direction

    '''def get_densities(self, tree, population):
        Nn = 20
        distances = tree.query(population.population_positions, k=Nn)[0]
        cache_key = hash(population.population_positions.tobytes())
        if cache_key in population._density_cache:
            return population._density_cache[cache_key]
        
        densities = (Nn-1)/distances[:, -1]
        population._density_cache[cache_key] = densities
        return densities'''
    
    '''def calculate_densest_position(self, tree, population):
        densities = self.get_densities(tree, population)
        return population.population_positions[np.argmax(densities)]'''
    def fnc(self, population):
        distances_to_prey = np.linalg.norm(population.inlier_positions - self.position, axis=1)
        predator_school_vectors = (population.inlier_positions - self.position) / distances_to_prey[:,np.newaxis]
        dot_products = np.dot(predator_school_vectors, self.direction)
        if np.any(dot_products < 0) and not np.all(dot_products < 0):
            self.minimum_distance_to_prey = min(distances_to_prey)
            #self.minimum_distances_to_prey = np.append(self.minimum_distances_to_prey, min(distances_to_prey))
        elif np.all(dot_products < 0):
            self.minimum_distance_to_prey = None
            return
        print(self.minimum_distance_to_prey)

    def update_predator(self, tree, population):
        self.fnc(population)
        self.calculate_direction(tree, population)
        if self.fixed_direction is not None:
            self.position += self.fixed_direction * self.speed
        else:
            self.position += self.direction * self.speed
