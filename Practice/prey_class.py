import numpy as np
import random
from parent_class import Parent
from environment_class import Environment
from predator_class import Predator

class Prey(Parent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.predator_separation_vector = np.zeros(2)

    def calculate_predator_separation_vector(self, birds):
        number_of_neighbours = 0
        for bird in birds:
            if isinstance(bird, Predator) and self.calculate_distance_to_birds(bird) <= self.neighbourhood:
                self.predator_separation_vector += (self.pos - bird.pos)/(self.calculate_distance_to_birds(bird))
                number_of_neighbours += 1
        if number_of_neighbours > 0:
            self.predator_separation_vector /= number_of_neighbours
        return self.predator_separation_vector
    
    def update_prey(self, birds, environment):
        '''update_prey is called every timestep for all (prey)agents.
        It calculates the relevant steering vectors for an agent and 
        updates their direction vector according to proximity to nearest
        neighbours.'''
        steering_vectors = self.calculate_steering_vector(birds, environment)

        v = np.random.normal(0, 0.2, (360,2))
        random_vector = random.choice(v)
        #random_vector /= np.linalg.norm(random_vector)

        if self.neighbours_in_repulsive_zone > 0:
            self.dir += steering_vectors[2]
        else:
            self.dir += (steering_vectors[0]
                        + steering_vectors[1])
            
        self.dir += steering_vectors[3] + random_vector  #regardless of neighbours agents will always steer
        self.dir /= np.linalg.norm(self.dir)             #away from walls and have a random component in their motion
        #print((f"Vectors: {steering_vectors}, Position: {self.pos}, Direction: {self.dir}"))
        self.pos += self.dir * self.speed