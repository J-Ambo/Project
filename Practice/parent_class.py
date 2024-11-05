import numpy as np
import random


class Parent:
    def __init__(self, x, y): 
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.dir /= np.linalg.norm(self.dir)  # Normalize the direction vector
        self.speed = 0.5                     # Speed of the bird
        self.neighbourhood = 8

        self.cohesion_factor, self.separation_factor, self.alignment_factor = 0.2, 0.2, 0.7
        self.alignment_vector, self.cohesion_vector, self.average_position_vector, self.separation_vector, self.wall_vector = np.zeros((5,2))

    def calculate_steering_vector(self, other_birds, environment):
        number_of_neighbours = 0
        for bird in other_birds:
            if self.calculate_distance_to_birds(bird) <= self.neighbourhood and bird != self: 
                self.alignment_vector += bird.dir
                self.average_position_vector += bird.pos
                self.separation_vector += (self.pos - bird.pos)/self.calculate_distance_to_birds(bird)
                number_of_neighbours += 1

            if np.linalg.norm(self.pos) >= environment.size * 0.5:
                self.wall_vector = -self.pos / np.linalg.norm(self.pos)
            else:
                self.wall_vector = np.zeros(2)

        if number_of_neighbours > 0:
            self.alignment_vector /= np.linalg.norm(self.alignment_vector)
            self.average_position_vector /= number_of_neighbours
            self.separation_vector /= np.linalg.norm(self.separation_vector)
            self.cohesion_vector = (self.average_position_vector - self.pos) / np.linalg.norm(self.average_position_vector - self.pos)
        else:
            self.alignment_vector = np.zeros(2)
            self.separation_vector = np.zeros(2)
            self.cohesion_vector = np.zeros(2)

        vectors = np.array([self.alignment_vector, self.cohesion_vector, self.separation_vector, self.wall_vector])
        return vectors
    
    def calculate_distance_to_birds(self, other_bird):
        distance = np.linalg.norm(self.pos - other_bird.pos,)
        if distance == 0:
            distance = 0.001
        return distance
    

'''  
b1 = Parent(0,0)
b2 = Parent(0,1)
b3 = Parent(1,0)
birds = [b1, b2 ,b3]
vectors = b1.calculate_steering_vector(birds)
print(vectors)
vectors[0]
'''