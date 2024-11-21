import numpy as np
from sklearn.neighbors import KDTree
import random

'''This script contains the Parent class, from which the Predator and Prey classes inherit key attributes and methods.'''
'''Abbreviations: rr = radius of repulsion;  ral = radius of alignment;  rat = radius of attraction;
                    rz = repulsion zone;  alz = alignmnet zone;  atz = attraction zone;
                    nn = number of neighbours'''

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

    def find_neighbours(self, other_agents):
        agent_positions = np.array([agent.position for agent in other_agents])
        tree = KDTree(agent_positions)

        distances, indices = tree.query(agent_positions, k=5)
        return distances, indices, tree
    
    def count_neighbours(self, tree, focus_point):

        indices_of_rr_neighbours = tree.query_radius(focus_point, r=self.radius_of_repulsion)
        indices_of_rz_neighbours = [item for item in indices_of_rr_neighbours[0] if item != 0]
        nn_within_rr = len(indices_of_rr_neighbours[0]) - 1  #minus one such that self is not counted as a neighbour.
        nn_in_rz = nn_within_rr #number of neighbours in the repulsion zone equals number of neighbours within the radius of repulsion.

        indices_of_ral_neighbours = tree.query_radius(focus_point, r=self.radius_of_alignment)
        indices_of_alz_neighbours = [item for item in indices_of_ral_neighbours[0] if item not in indices_of_rr_neighbours[0]]
        nn_within_ral = len(indices_of_ral_neighbours[0]) - 1
        nn_in_alz = nn_within_ral - nn_within_rr

        indices_of_rat_neighbours = tree.query_radius(focus_point, r=self.radius_of_attraction)
        indices_of_atz_neighbours = [item for item in indices_of_rat_neighbours[0] if item not in indices_of_ral_neighbours[0]]
        nn_within_rat = len(indices_of_rat_neighbours[0]) - 1
        nn_in_atz = nn_within_rat - nn_within_ral
        return nn_in_rz, nn_in_alz, nn_in_atz, indices_of_rz_neighbours, indices_of_alz_neighbours, indices_of_atz_neighbours




    def calculate_steering_vectors(self, other_agents, environment, distances, indices, tree):
        (self.neighbours_in_repulsive_zone,
        self.neighbours_in_alignment_zone, 
        self.neighbours_in_attraction_zone) = 0, 0, 0 

        (self.separation_vector, 
         self.alignment_vector, 
         self.cohesion_vector, 
         self.wall_vector) = np.zeros((4,2))
        
        steering_vectors = np.zeros((len(other_agents), 4, 2))

        for i in range(len(indices)):
            focus_point = other_agents[i].position.reshape(1,-1)
            distance_from_origin = np.linalg.norm(other_agents[i].position)
            distance_from_boundary = environment.radius - distance_from_origin
            n_rz, n_alz, n_atz, i_rz, i_alz, i_atz = self.count_neighbours(tree, focus_point)

            if n_rz > 0 and i not in i_rz:
                separation_distances_rz = {n: distances[i, np.where(indices[i] == n)[0][0]] for n in i_rz if n != i}
                print(separation_distances_rz)
                other_agents[i].separation_vector = -sum((other_agents[n].position - other_agents[i].position)/separation_distances_rz[n] for n in i_rz)
            else:
                other_agents[i].separation_vector = np.array([0,0])
            #print(other_agents[i].separation_vector.shape)
            #print(other_agents[i].separation_vector)

            if n_alz > 0:
                other_agents[i].alignment_vector = sum(other_agents[n].direction for n in i_alz)
            else:
                other_agents[i].alignment_vector = np.array([0,0])

            if n_atz > 0 and i not in i_atz:
                separation_distances_atz = {n: distances[i, np.where(indices[i] == n)[0][0]] for n in i_atz if n != i}
                other_agents[i].attraction_vector = sum((other_agents[n].position - other_agents[i].position)/separation_distances_atz[n] for n in i_atz)
            else:
                other_agents[i].attraction_vector = np.array([0,0])

            if environment.radius * 0.9 < distance_from_origin < environment.radius:
                other_agents[i].wall_vector = -self.position * np.exp(-distance_from_boundary)
            else:
                other_agents[i].wall_vector = np.array([0,0])

            steering_vectors[i] = np.array([other_agents[i].separation_vector, other_agents[i].alignment_vector, other_agents[i].attraction_vector, other_agents[i].wall_vector])
        return steering_vectors
            #neighbour_indices = indices[i,1:]
            # neighbour_distances = distances[i,1:]





        







        '''for agent in other_agents:
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
        return vectors'''


