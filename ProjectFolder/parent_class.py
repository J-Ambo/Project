import numpy as np

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''
'''Abbreviations: rr = radius of repulsion;  ral = radius of alignment;  rat = radius of attraction;
                    rz = repulsion zone;  alz = alignmnet zone;  atz = attraction zone;
                    nn = number of neighbours'''

class Parent:
    rr = 1.0  #radius of repulsion
    ral = 1  #radius of alignment
    rat = 15
    speed = 3
    perception_angle = np.deg2rad(270)
    maximal_turning_angle = np.deg2rad(40) #* 0.1    #0.1s is the time step

    def __init__(self, x, y, z, dimensions): 
        self.position = np.array([x, y, z])   

        self.direction = np.random.uniform(-1,1, 3)
        normal = np.linalg.norm(self.direction)
        self.direction /= normal

        self.body_length = 1
        self.speed =  self.__class__.speed  #np.clip(np.random.normal(loc=0.5, scale=0.1), 0.45, 0.55)     #np.random.choice(np.linspace(0.5, 1, 5))
        self.perception_angle = self.__class__.perception_angle
        self.minimum_turning_radius = 0.2 * self.body_length
        self.maximal_turning_angle = self.__class__.maximal_turning_angle #np.arcsin(self.speed / (2 * self.minimum_turning_radius))

        self.maximal_rotation_matrix = np.array([[np.cos(self.maximal_turning_angle), -np.sin(self.maximal_turning_angle)],
                                               [np.sin(self.maximal_turning_angle), np.cos(self.maximal_turning_angle)]])

        (self.radius_of_repulsion,
        self.radius_of_alignment, 
        self.radius_of_attraction) = self.__class__.rr, self.__class__.ral, self.__class__.rat

        (self.repulsion_vector, 
         self.alignment_vector, 
         self.attraction_vector, 
         self.wall_vector) = np.zeros((4,dimensions))
        
    @classmethod
    def increment_ral(cls, increment):
        cls.ral += increment

    @classmethod
    def increment_rat(cls, increment):
        cls.rat += increment



        



