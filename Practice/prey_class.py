import numpy as np
from parent_class import Parent
from environment_class import Environment
from predator_class import Predator

class Prey(Parent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.predator_separation_vector = np.zeros(2)

    def calculate_predator_separation_vector(self, birds):
        num_neighbours = 0
        for bird in birds:
            if isinstance(bird, Predator) and self.calculate_distance_to_birds(bird) <= self.neighbourhood:
                self.predator_separation_vector += (self.pos - bird.pos)/(self.calculate_distance_to_birds(bird))
                num_neighbours += 1
        if num_neighbours > 0:
            self.predator_separation_vector /= num_neighbours
        return self.predator_separation_vector
    
    def update_prey(self, birds, environment):
        self.dir += (self.alignment_factor * self.calculate_steering_vector(birds)[0]) + (self.cohesion_factor * self.calculate_steering_vector(birds)[1])
        + (self.separation_factor * self.calculate_steering_vector(birds)[2]) + (self.calculate_predator_separation_vector(birds))
        self.dir /= np.linalg.norm(self.dir)   
        if self.point_is_out_of_bounds(self.pos[0], environment):
            self.pos[0] = self.apply_boudary_condition(self.pos[0], environment)
            self.dir[0] *= -1
        else:
            self.pos[0] += self.dir[0] * self.speed

        if self.point_is_out_of_bounds(self.pos[1], environment):
            self.pos[1] = self.apply_boudary_condition(self.pos[1], environment)
            self.dir[1] *= -1
        else:
            self.pos[1] += self.dir[1] * self.speed
