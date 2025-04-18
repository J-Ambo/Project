import numpy as np
from AgentClasses.parent_class import Parent
from AgentClasses.environment_class import Environment
from AgentClasses.population_class import Population
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
        self.fov = np.deg2rad(70)
        self.previous_neighbours = np.array([])
        self.previous_targets = np.array([])
        self.attack_number = 0
        self.neighbour_densities = None
        self.success = None

    def find_neighbours(self, tree, population, dov):
        neighbours, distances = tree.query_radius([self.position], dov, return_distance=True)
        self_mask = (distances[0] != 0)
        
        neighbours = neighbours[0][self_mask]
        distances = distances[0][self_mask]

        if neighbours.size == 0:
            #print('early exit')
            return np.array([]), np.array([])
        
        neighbour_positions = population.population_positions[neighbours]
        directions_to_neighbours = (neighbour_positions - self.position)
        np.divide(directions_to_neighbours, distances[:,np.newaxis], out=directions_to_neighbours)

        dot_products = np.dot(directions_to_neighbours, self.direction)
        dot_products = np.round(dot_products, 3)

        angles_to_neighbours = np.arccos(dot_products)
        mask = angles_to_neighbours <= self.fov/2
        visible_indices = neighbours[mask]
        visible_distances = distances[mask]

        return visible_indices, visible_distances      #population.all_positions[visible_indices]

    def calculate_direction(self, population):
        vector_to_com = population.average_school_position - self.position
        distance_to_com = np.linalg.norm(vector_to_com) 
        
        if distance_to_com > 5 and self.skip == False:  
            self.direction = vector_to_com / distance_to_com

        elif distance_to_com <= 5 or self.skip == True:
            self.skip = True
            self.fixed_direction = self.direction

    def fnc(self, tree, population):
        neighbours_indices, distances = self.find_neighbours(tree, population, 7)

        msk = distances<2
        targets = neighbours_indices[msk]
        if targets.size != 0:
            target_mask = distances[msk] == min(distances[msk])
            self.attack_number += 1
          #  print(targets[target_mask])
    
        if neighbours_indices.size == 0:
            self.minimum_distance_to_prey = None
            self.neighbour_densities = None
            self.success = None
        else:
            self.minimum_distance_to_prey = min(distances)
            self.neighbour_densities = np.nanmean(population.population_densities[neighbours_indices])
            self.success = 1/(len(neighbours_indices))  #*self.minimum_distance_to_prey)    #*self.neighbour_densities
           # print(self.success)

        self.previous_neighbours = neighbours_indices
       # self.previous_targets = targets

    def calculate_angles(self, positions, directions):
       # neighbours_indices, distances = self.find_neighbours(tree, population, 7)
        predator_school_vectors = (positions - self.position)
        norms = np.linalg.norm(predator_school_vectors, axis=1)
        #print(norms)
        predator_school_vectors /= norms[:, np.newaxis]
        dot_products = np.dot(predator_school_vectors, self.direction)

        if np.any(dot_products <= 0) and np.any(dot_products >= 0):
            self.attack = True
            predator_prey_angles = np.rad2deg(np.arccos(dot_products))
        ##    print('ppa',predator_prey_angles)
            self.predator_prey_angles = predator_prey_angles  #np.append(self.predator_prey_angles, predator_prey_angles)

            dt_prod = np.einsum('ij, ij->i', -predator_school_vectors, directions)
            prey_orientation = np.rad2deg(np.arccos(dt_prod))
        ##    print('po',prey_orientation)
           # print(np.mean(prey_angles))
            self.prey_orientation = prey_orientation
           # print(self.predator_prey_angles)

            self.distance2prey = norms #np.mean(norms)
            #print(norms)
        
        else:
            self.attack = False

    def update_predator(self, tree, population):
       # self.fnc(tree, population)
        self.calculate_direction(population)
        if self.fixed_direction is not None:
            self.position += self.fixed_direction * self.speed * 0.1
        else:
            self.position += self.direction * self.speed * 0.1

        population.population_positions[-1] = self.position
        population.population_directions[-1] = self.direction
