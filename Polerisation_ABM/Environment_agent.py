import numpy as np

class Environment:
    def __init__(self, size):
        self.size = size
    
    def create_environment(self):
        self.environment = np.zeros((self.size, self.size))
        return self.environment
