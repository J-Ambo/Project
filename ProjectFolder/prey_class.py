import numpy as np
import random
from parent_class import Parent
from environment_class import Environment
from predator_class import Predator

'''This script contains the Prey class, which inherits from the Parent class, and is used to create instances of prey agents.
Differs from the Predator class in that it has a method to steer away from predators, and update its directionection vector accordingly.'''

class Prey(Parent):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.predator_separation_vector = np.zeros(2)

    def calculate_predator_separation_vector(self, agents):
        number_of_neighbours = 0
        for agent in agents:
            if isinstance(agent, Predator) and self.calculate_distance_to_agents(agent) <= self.neighbourhood:
                self.predator_separation_vector += (self.position - agent.position)/(self.calculate_distance_to_agents(agent))
                number_of_neighbours += 1
        if number_of_neighbours > 0:
            self.predator_separation_vector /= number_of_neighbours
        return self.predator_separation_vector
    
    def update_prey(self, agents, environment):
        '''update_prey is called every timestep for all (prey)agents.
        It calculates the relevant steering vectors for an agent and 
        updates their direction vector according to proximity to nearest
        neighbours.'''
        steering_vectors = self.calculate_steering_vector(agents, environment)

        v = np.random.normal(0, 0.5, (360,2))
        random_vector = random.choice(v)
        #random_vector /= np.linalg.norm(random_vector)

        if self.neighbours_in_repulsive_zone > 0:
            self.direction += steering_vectors[0]
        else:
            self.direction += steering_vectors[1] + steering_vectors[2]
            
        self.direction += steering_vectors[3] #+ random_vector     #regardless of neighbours, agents will always steer
        self.direction /= np.linalg.norm(self.direction)          #away from walls and have a random component in their motion     

        print((f"Vectors: {steering_vectors}, Position:{self.position}, direction: {self.direction}"))

        self.position += self.direction * self.speed
        self.position = np.array([round(self.position[0], 2), round(self.position[1], 2)])