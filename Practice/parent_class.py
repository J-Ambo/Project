import numpy as np
import random


class Parent:
    def __init__(self, x, y): 
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.dir /= np.linalg.norm(self.dir)  # Normalize the direction vector
        self.speed = 0.5                     # Speed of the bird
        self.neighbourhood = 20

        self.cohesion_factor, self.separation_factor, self.alignment_factor = 0.05, 0.05, 0.05
        self.alignment_vector, self.cohesion_vector, self.average_position_vector, self.separation_vector = np.zeros((4,2))

    def calculate_steering_vector(self, other_birds):
        number_of_neighbours = 0
        for bird in other_birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                self.alignment_vector += bird.dir
                self.average_position_vector += bird.pos
                self.separation_vector += (self.pos - bird.pos)/self.calculate_distance_to_birds(bird)
                number_of_neighbours += 1
        if number_of_neighbours > 0:
            self.alignment_vector /= number_of_neighbours
            self.average_position_vector /= number_of_neighbours
            self.separation_vector /= number_of_neighbours
            self.cohesion_vector = self.average_position_vector - self.pos
        return self.alignment_vector, self.cohesion_vector, self.separation_vector
    
    def calculate_distance_to_birds(self, other_bird):
        #print(other_bird.pos)
        distance = np.linalg.norm(self.pos - other_bird.pos,)
        if distance == 0:
            distance = 0.001
        return distance

    def point_is_out_of_bounds(self, coordinate, environment):
        if coordinate >= environment.size or coordinate <= 0:
            return True
        return False

    def apply_boundary_condition(self, coordinate, environment):
        if coordinate <= 0:
            coordinate = 0 + 0.01 * environment.size
        elif coordinate >= environment.size:
            coordinate = 0.99 * environment.size
        return coordinate