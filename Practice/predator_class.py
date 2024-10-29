import numpy as np
from parent_class import Parent
from environment_class import Environment


class Predator(Parent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.speed = 0.9*self.speed
        
    def update_predator(self, other_birds, environment):
        self.dir += (self.alignment_factor * self.calculate_steering_vector(other_birds)[0]) + (self.cohesion_factor * self.calculate_steering_vector(other_birds)[1]) + (self.separation_factor * -self.calculate_steering_vector(other_birds)[2])
        self.dir /= np.linalg.norm(self.dir) 
        if Parent.point_is_out_of_bounds(Parent, self.pos[0], environment):
            self.pos[0] = Parent.apply_boudary_condition(Parent, self.pos[0], environment)
            self.dir[0] *= -1
        else:
            self.pos[0] += self.dir[0] * self.speed

        if self.point_is_out_of_bounds(self.pos[1], environment):
            self.pos[1] = self.apply_boudary_condition(self.pos[1], environment)
            self.dir[1] *= -1
        else:
            self.pos[1] += self.dir[1] * self.speed