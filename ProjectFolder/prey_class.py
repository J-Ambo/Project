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
    
    def update_position(self, steering_vectors):
        '''update_position is called every timestep for all (prey)agents.
        It calculates the relevant steering vectors for an agent and 
        updates their direction vector according to proximity to nearest
        neighbours.'''

        target_direction = np.zeros(2)
        random_angle = np.random.normal(0, np.pi/16)
        random_angle = np.clip(random_angle, -np.pi/16, np.pi/16)
        random_rotation_matrix = np.array([[np.cos(random_angle), -np.sin(random_angle)], [np.sin(random_angle), np.cos(random_angle)]])
        threshold_rotation_matrix = np.array([[np.cos(np.pi/8), -np.sin(np.pi/8)], [np.sin(np.pi/8), np.cos(np.pi/8)]])

        if self.neighbours_in_repulsive_zone > 0:
            target_direction += steering_vectors[0] + 0.1*(steering_vectors[1] + steering_vectors[2])
        elif self.neighbours_in_alignment_zone > 0 or self.neighbours_in_attraction_zone > 0:
            target_direction += steering_vectors[1] + steering_vectors[2]
        else:
            target_direction += self.direction
        
        target_direction += steering_vectors[3]       #regardless of neighbours, agents will always steer away from walls.    
    
        target_direction /= np.linalg.norm(target_direction)
        angle_to_target_direction = np.arccos(np.clip(np.dot(self.direction, target_direction), -1.0, 1.0))
        
        if angle_to_target_direction < np.pi / 8:  # threshold angle
            self.direction = target_direction
        else:
            z_cross_component = self.direction[0] * target_direction[1] - self.direction[1] * target_direction[0]
            if z_cross_component > 0:
                self.direction = np.dot(threshold_rotation_matrix, self.direction)
            else:
                self.direction = np.dot(threshold_rotation_matrix.T, self.direction)
        
        

        self.direction = np.dot(random_rotation_matrix, self.direction)        #rotate the diretion vector by a random angle.
        self.direction /= np.linalg.norm(self.direction) 
        
        self.position += self.direction * self.speed
        self.position = np.array([round(self.position[0], 2), round(self.position[1], 2)])




