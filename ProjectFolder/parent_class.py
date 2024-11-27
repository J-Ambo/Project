import numpy as np
from sklearn.neighbors import KDTree
import random

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''
'''Abbreviations: rr = radius of repulsion;  ral = radius of alignment;  rat = radius of attraction;
                    rz = repulsion zone;  alz = alignmnet zone;  atz = attraction zone;
                    nn = number of neighbours'''

class Parent:
    def __init__(self, x, y, z, dimensions): 
        self.position = np.array([x, y, z])  
        self.previous_position = np.zeros(dimensions) 

        self.direction = np.random.rand(dimensions)
        self.direction /= np.linalg.norm(self.direction)
        self.previous_direction = self.direction

        self.body_length = 1
        self.speed = np.clip(np.random.normal(loc=0.5, scale=0.1), 0.45, 0.55)     #np.random.choice(np.linspace(0.5, 1, 5))
        self.perception_angle = np.deg2rad(270)
        self.minimum_turning_radius = 0.2 * self.body_length
        self.maximal_turning_angle = np.deg2rad(60)  #np.arcsin(self.speed / (2 * self.minimum_turning_radius))

        self.maximal_rotation_matrix = np.array([[np.cos(self.maximal_turning_angle), -np.sin(self.maximal_turning_angle)],
                                               [np.sin(self.maximal_turning_angle), np.cos(self.maximal_turning_angle)]])

        (self.radius_of_repulsion,
        self.radius_of_alignment, 
        self.radius_of_attraction) = 1, 5, 20
        
        (self.repulsion_vector, 
         self.alignment_vector, 
         self.attraction_vector, 
         self.wall_vector) = np.zeros((4,2))


