import numpy as np
import random

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''

class Parent:
    def __init__(self, x, y): 
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.dir /= np.linalg.norm(self.dir)  # Normalize the direction vector
        self.speed = np.random.choice(np.linspace(0.4, 0.5, 5))                     # Speed of the bird
        (self.radius_of_repulsion,
        self.radius_of_alignment, 
        self.radius_of_attraction)  = 1, 4, 10
        
        (self.neighbours_in_repulsive_zone, 
         self.neighbours_in_alignment_zone, 
         self.neighbours_in_attraction_zone) = 0, 0, 0
        
    def calculate_distance_to_birds(self, other_bird):
        distance = np.linalg.norm(self.pos - other_bird.pos,)
        if distance == 0:
            distance = 0.001
        return distance

    def calculate_steering_vector(self, other_birds, environment):
        (self.neighbours_in_repulsive_zone,
        self.neighbours_in_alignment_zone, 
        self.neighbours_in_attraction_zone) = 0, 0, 0 

        (self.alignment_vector, 
         self.cohesion_vector, 
         self.average_position_vector, 
         self.separation_vector, 
         self.wall_vector) = np.zeros((5,2))

        for bird in other_birds:
            if bird == self:
                continue

            if self.calculate_distance_to_birds(bird) <= self.radius_of_repulsion:
                self.separation_vector += -(bird.pos - self.pos)/np.linalg.norm(bird.pos - self.pos)
                self.neighbours_in_repulsive_zone += 1

            if self.radius_of_repulsion <= self.calculate_distance_to_birds(bird) <= self.radius_of_alignment:
                self.alignment_vector += bird.dir/np.linalg.norm(bird.dir)
                self.neighbours_in_alignment_zone += 1

            if self.radius_of_alignment <= self.calculate_distance_to_birds(bird) <= self.radius_of_attraction:
                self.average_position_vector += bird.pos
                self.neighbours_in_attraction_zone += 1

            if np.linalg.norm(self.pos) >= (environment.size*0.5) - self.radius_of_repulsion*0.5:
                self.wall_vector = -self.pos#/np.linalg.norm(self.pos)

        vectors = np.array([self.alignment_vector, self.cohesion_vector, self.separation_vector, self.wall_vector])
        return vectors