import numpy as np
import random
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from prey_class import Prey
from environment_class import Environment
from population_class import Population

env = Environment(10)
#pop = Population()

import numpy as np
import random
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from prey_class import Prey
from environment_class import Environment
from population_class import Population

# Define the population size
population_size = 3
population_directions = np.array([[1, 0], [0, 1], [-1, 0]])
# Generate random angles for the population
random_angles = np.array([np.pi, 0, np.pi/2])#np.random.normal(0, 0.2, population_size)
random_rotation_matrices = np.zeros((population_size, 2, 2))
c = np.round(np.cos(random_angles), 2)
s = np.round(np.sin(random_angles), 2)
random_rotation_matrices[:, 0, 0] = c
random_rotation_matrices[:, 0, 1] = -s
random_rotation_matrices[:, 1, 0] = s
random_rotation_matrices[:, 1, 1] = c



print(random_rotation_matrices)
# Transpose individual matrices in random_rotation_matrices
transposed_matrices = np.transpose(random_rotation_matrices, axes=(0, 2, 1))
print(transposed_matrices)
result = np.matmul(random_rotation_matrices, population_directions[:, :, np.newaxis])
print(result[:,:, 0])

# Example of broadcasting matrix multiplication
A = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 2]]])
B = np.array([[1, 0], [1, 1]])[:, :, np.newaxis]
# Broadcasting matrix multiplication
result = A@B #np.matmul(A, B)
result = result[:, :, 0]
#print(result)