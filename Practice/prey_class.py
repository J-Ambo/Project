import numpy as np
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
        steering_vectors = self.calculate_steering_vector(birds, environment)

        self.dir += (self.alignment_factor * steering_vectors[0]
                    + self.cohesion_factor * steering_vectors[1]
                    + self.separation_factor * steering_vectors[2]
                    + steering_vectors[3] * 8
                    + self.calculate_predator_separation_vector(birds))
        self.dir /= np.linalg.norm(self.dir)  
        #print((f"Vectors: {steering_vectors}, Position: {self.pos}, Direction: {self.dir}"))
 
        self.pos += self.dir * self.speed
