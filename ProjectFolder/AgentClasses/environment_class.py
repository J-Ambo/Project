import numpy as np
'''This script contains the Environment class. This class is used to create an environment instance'''

class Environment:
    def __init__(self, size, dimensions):
        self.radius = size
        self.dimension = dimensions