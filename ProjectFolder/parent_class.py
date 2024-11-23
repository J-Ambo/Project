import numpy as np
from sklearn.neighbors import KDTree
import random

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''
'''Abbreviations: rr = radius of repulsion;  ral = radius of alignment;  rat = radius of attraction;
                    rz = repulsion zone;  alz = alignmnet zone;  atz = attraction zone;
                    nn = number of neighbours'''

class Parent:
    def __init__(self, x, y): 
        self.position = np.array([x, y])  
        self.previous_position = np.zeros(2) 

        self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.direction /= np.linalg.norm(self.direction)
        self.previous_direction = self.direction

        self.speed = np.clip(np.random.normal(loc=0.5, scale=0.1), 0.45, 0.55)     #np.random.choice(np.linspace(0.5, 1, 5))
        self.perception_angle = np.deg2rad(270)

        (self.radius_of_repulsion,
        self.radius_of_alignment, 
        self.radius_of_attraction) = 1, 4, 50
        
        (self.neighbours_in_repulsive_zone, 
         self.neighbours_in_alignment_zone, 
         self.neighbours_in_attraction_zone) = 0, 0, 0
        
        (self.repulsion_vector, 
         self.alignment_vector, 
         self.attraction_vector, 
         self.wall_vector) = np.zeros((4,2))


