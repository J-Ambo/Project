import numpy as np
from sklearn.neighbors import KDTree
import random

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''

class Parent:
    def __init__(self, x, y): 
        self.position = np.array([x, y])   

        self.direction = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.direction /= np.linalg.norm(self.direction)

        self.speed = np.clip(np.random.normal(loc=0.5, scale=0.1), 0.45, 0.55)     #np.random.choice(np.linspace(0.5, 1, 5))
        self.perception_angle = np.deg2rad(270)

        (self.radius_of_repulsion,
        self.radius_of_alignment, 
        self.radius_of_attraction) = 4, 9, 11
        
        (self.neighbours_in_repulsive_zone, 
         self.neighbours_in_alignment_zone, 
         self.neighbours_in_attraction_zone) = 0, 0, 0
        
    def calculate_distance_to_agent(self, other_agent):
        distance = np.linalg.norm(self.position - other_agent.position)
        return max(distance, 0.01)
    
    def calculate_nearest_neighbour_distance(self, other_agents):
        distances = [self.calculate_distance_to_agent(agent) for agent in other_agents]
        average_distance = np.mean(distances)
        return min(distances), 

    def calcualte_neighbours(self, other_agents):
        agent_positions = np.array([agent.position for agent in other_agents])
        tree = KDTree(agent_positions)

        query_point = self.position.reshape(1, -1)
        indices_repulsion_radius_neighbours = tree.query_radius(query_point, r=self.radius_of_repulsion)
        indices_alignment_radius_neighbours = tree.query_radius(query_point, r=self.radius_of_alignment)
        indices_attraction_radius_neighbours = tree.query_radius(query_point, r=self.radius_of_attraction)

        distances, indices = tree.query(agent_positions, k=3)
        return distances, indices, indices_repulsion_radius_neighbours

    def calculate_steering_vector(self, other_agents, environment):
        (self.neighbours_in_repulsive_zone,
        self.neighbours_in_alignment_zone, 
        self.neighbours_in_attraction_zone) = 0, 0, 0 

        (self.separation_vector, 
         self.alignment_vector, 
         self.cohesion_vector, 
         self.wall_vector) = np.zeros((4,2))

        for agent in other_agents:

            distance_between_agents = self.calculate_distance_to_agent(agent)
            direction_to_other_agent = (agent.position - self.position) / distance_between_agents
            angle_to_other_agent = np.arccos(np.dot(self.direction, direction_to_other_agent))

            if agent == self:
                distance_from_origin = np.linalg.norm(self.position)
                distance_from_boundary = environment.radius - np.linalg.norm(self.position)
                
                if environment.radius * 0.9 < distance_from_origin < environment.radius:
                    self.wall_vector = -self.position * np.exp(-distance_from_boundary)
                continue

            if True:    
                if angle_to_other_agent > self.perception_angle/2:
                    continue

            if distance_between_agents <= self.radius_of_repulsion:
                self.neighbours_in_repulsive_zone += 1
                self.separation_vector += -(agent.position - self.position) / distance_between_agents
                
            if self.radius_of_repulsion < distance_between_agents <= self.radius_of_alignment:
                self.neighbours_in_alignment_zone += 1
                self.alignment_vector += agent.direction / np.linalg.norm(agent.direction)
                
            if self.radius_of_alignment < distance_between_agents <= self.radius_of_attraction:
                self.neighbours_in_attraction_zone += 1
                self.cohesion_vector += (agent.position - self.position) / distance_between_agents

        vectors = np.array([self.separation_vector, 
                            self.alignment_vector, 
                            self.cohesion_vector, 
                            self.wall_vector])
        return vectors


