import numpy as np
import random
from parent_class import Parent
from environment_class import Environment
from predator_class import Predator

'''This script contains the Prey class, which inherits from the Parent class, and is used to create instances of prey agents.
Differs from the Predator class in that it has a method to steer away from predators, and update its directionection vector accordingly.

#####  ^^DEFUNCT  #####

The prey agents are now initialised directly from the Parent class, and their steering methods from the Population class'''


class Prey(Parent):
    def __init__(self, x, y, z, dimensions):
        super().__init__(x, y, z, dimensions)





