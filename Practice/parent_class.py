import numpy as np
import random


class Parent:
    def __init__(self, x, y): 
        self.pos = np.array([x, y])         # Position of the bird
        self.dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.dir /= np.linalg.norm(self.dir)  # Normalize the direction vector
        self.speed = 0.5                     # Speed of the bird
        self.radius_of_repulsion = 1
        self.radius_of_alignment = 4
        self.radius_of_attraction = 6

        '''(self.cohesion_factor, 
         self.separation_factor, 
         self.alignment_factor) = 1, 1, 1'''
        
        (self.neighbours_in_repulsive_zone, 
         self.neighbours_in_alignment_zone, 
         self.neighbours_in_attraction_zone) = 0, 0, 0
        

    def calculate_steering_vector(self, other_birds, environment):
        neighbours_in_repulsive_zone = 0
        neighbours_in_alignment_zone = 0
        neighbours_in_attraction_zone = 0
        self.alignment_vector, self.cohesion_vector, self.average_position_vector, self.separation_vector, self.wall_vector = np.zeros((5,2))

        for bird in other_birds:

            if self.calculate_distance_to_birds(bird) <= self.radius_of_repulsion and bird != self:
                self.separation_vector += -(bird.pos - self.pos)/np.linalg.norm(bird.pos - self.pos)
                neighbours_in_repulsive_zone += 1
            '''else:
                self.separation_vector = np.zeros(2)
                neighbours_in_repulsive_zone = 0'''

            if self.radius_of_repulsion <= self.calculate_distance_to_birds(bird) <= self.radius_of_alignment and bird != self:
                self.alignment_vector += bird.dir/np.linalg.norm(bird.dir)
                neighbours_in_alignment_zone += 1
            '''else:
                self.alignment_vector = np.zeros(2)
                neighbours_in_alignment_zone = 0'''

            if self.radius_of_alignment <= self.calculate_distance_to_birds(bird) <= self.radius_of_attraction and bird != self:
                self.average_position_vector += bird.pos
                neighbours_in_attraction_zone += 1
            '''else:
                self.average_position_vector = np.zeros(2)
                neighbours_in_attraction_zone = 0'''

            if np.linalg.norm(self.pos) >= (environment.size*0.5) - self.radius_of_repulsion*0.5:
                self.wall_vector = -self.pos
            else:
                self.wall_vector = np.zeros(2)

        '''if neighbours_in_attraction_zone > 0:
            self.average_position_vector /= neighbours_in_attraction_zone
            self.cohesion_vector = (self.average_position_vector - self.pos) / np.linalg.norm(self.average_position_vector - self.pos)

        if neighbours_in_alignment_zone > 0:
             self.alignment_vector /= np.linalg.norm(self.alignment_vector)
        
        if neighbours_in_repulsive_zone > 0:
            self.separation_vector /= np.linalg.norm(self.separation_vector)'''
        
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