import numpy as np

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''
'''Abbreviations: rr = radius of repulsion;  ral = radius of alignment;  rat = radius of attraction;
                    rz = repulsion zone;  alz = alignmnet zone;  atz = attraction zone;
                    nn = number of neighbours'''

class Parent:
    rr = 1.0  #radius of repulsion
    ral = None  #radius of alignment
    rat = None
    speed = None
    perception_angle = None
    maximal_turning_angle = None
    evasion_angle = None

    def __init__(self, x, y, z): 
        self.position = np.array([x, y, z], dtype=np.float64)   

        self.direction = np.random.uniform(-1,1, 3)
        self.direction /= np.linalg.norm(self.direction)

        self.speed =  self.__class__.speed  
        self.perception_angle = self.__class__.perception_angle
        #self.body_length = 1
        #self.minimum_turning_radius = 0.2 * self.body_length
        self.maximal_turning_angle = self.__class__.maximal_turning_angle #np.arcsin(self.speed / (2 * self.minimum_turning_radius))

    @classmethod
    def increment_ral(cls, increment):
        cls.ral += increment

    @classmethod
    def increment_rat(cls, increment):
        cls.rat += increment

    

        



