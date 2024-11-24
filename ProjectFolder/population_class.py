import numpy as np 
import random
from sklearn.neighbors import KDTree
from parent_class import Parent
from prey_class import Prey
from environment_class import Environment
import timeit


class Population:
    def __init__(self, population_size, number_of_neighbours, environment):
        self.population_size = population_size
        self.n_neighbours = number_of_neighbours
        self.population_array = np.zeros(population_size, dtype=object)

        self.population_positions = np.zeros((population_size, 2))
        self.population_directions = np.zeros((population_size, 2))
        self.population_speeds = np.zeros(population_size)
        self.population_angular_momenta = np.zeros(population_size)
                                              
        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter

        for n in range(population_size):
            r = random.uniform(0, environment.radius)
            theta = random.uniform(0, 2*np.pi)

            x = r*np.cos(theta)*0.5
            y = r*np.sin(theta)*0.5
            agent = Prey(x,y)
            self.population_array[n] = agent
            self.population_positions[n] = agent.position

    def find_neighbours(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        distances, indices = tree.query(all_positions, k=self.n_neighbours+1)
        return distances, indices, tree

    def calculate_vectors(self, environment):
        distances, indices, tree = self.find_neighbours()
        vectors = np.zeros((self.population_size, 4, 2))
        
        for n in range(self.population_size):
            nearest_neighbour_distances = distances[n,1:]
            nearest_neighbour_indices = indices[n,1:]
            agent_n = self.population_array[n]

            distance_from_origin = np.linalg.norm(agent_n.position)
            distance_from_boundary = environment.radius - distance_from_origin

            agent_n.repulsion_vector = np.zeros(2)
            agent_n.alignment_vector = np.zeros(2)
            agent_n.attraction_vector = np.zeros(2)
            agent_n.wall_vector = np.zeros(2)

            for i in range(self.n_neighbours):
                agent_i = self.population_array[nearest_neighbour_indices[i]]
                agent_i_distance = nearest_neighbour_distances[i]

                direction_to_agent_i = (agent_i.position - agent_n.position) / agent_i_distance
                angle_to_agent_i = np.arccos(np.dot(agent_n.direction, direction_to_agent_i))

                if angle_to_agent_i > agent_n.perception_angle/2:
                    continue

                if environment.radius * 0.9 < distance_from_origin < environment.radius:
                    agent_n.wall_vector += -agent_n.position * np.exp(-distance_from_boundary)
                elif distance_from_origin >= environment.radius:
                    agent_n.wall_vector += -agent_n.position
                    continue

                if agent_i_distance < agent_n.radius_of_repulsion:
                    agent_n.repulsion_vector += -(agent_i.position - agent_n.position)/agent_i_distance
                    continue
 
                if agent_n.radius_of_repulsion <= agent_i_distance < agent_n.radius_of_alignment:
                    agent_n.alignment_vector += agent_i.direction

                if agent_n.radius_of_alignment <= agent_i_distance < agent_n.radius_of_attraction:
                    agent_n.attraction_vector += (agent_i.position - agent_n.position)/agent_i_distance

            vectors[n] = np.array([agent_n.repulsion_vector, agent_n.alignment_vector, agent_n.attraction_vector, agent_n.wall_vector])  
        return vectors

    def update_positions(self, environment):
        steering_vectors = self.calculate_vectors(environment)
        average_position = np.mean(self.population_positions, axis=0)

        for n in range(self.population_size):
            agent_n = self.population_array[n]
            agent_n_vectors = steering_vectors[n]
            agent_n_target_direction = np.zeros(2)
            average_position_to_agent_n = (agent_n.position - average_position) / np.linalg.norm(agent_n.position - average_position)

            random_angle = np.random.normal(0, 0.2)
            random_rotation_matrix = np.array([[np.cos(random_angle), -np.sin(random_angle)], [np.sin(random_angle), np.cos(random_angle)]])
            threshold_rotation_matrix = np.array([[np.cos(agent_n.maximal_turning_angle), -np.sin(agent_n.maximal_turning_angle)], [np.sin(agent_n.maximal_turning_angle), np.cos(agent_n.maximal_turning_angle)]])

            sum_of_vectors = np.sum(agent_n_vectors, axis=0)
            if np.all(sum_of_vectors == 0):
                agent_n_target_direction += agent_n.direction
            else:
                agent_n_target_direction += sum_of_vectors
            
            agent_n_target_direction /= np.linalg.norm(agent_n_target_direction)
            angle_to_target_direction = np.arccos(np.clip(np.dot(agent_n.direction, agent_n_target_direction), -1.0, 1.0))

            if angle_to_target_direction < agent_n.maximal_turning_angle:  # threshold angle
                agent_n.direction = agent_n_target_direction
            else:
                z_cross_component = agent_n.direction[0] * agent_n_target_direction[1] - agent_n.direction[1] * agent_n_target_direction[0]
                if z_cross_component > 0:
                    agent_n.direction = np.dot(threshold_rotation_matrix, agent_n.direction)
                else:
                    agent_n.direction = np.dot(threshold_rotation_matrix.T, agent_n.direction)
            
            agent_n.direction = np.dot(random_rotation_matrix, agent_n.direction)
            agent_n.direction /= np.linalg.norm(agent_n.direction)
            agent_n.previous_direction = agent_n.direction
            agent_n.position += agent_n.speed * agent_n.direction
            agent_n_angular_momentum = np.cross(average_position_to_agent_n, agent_n.direction)

            self.population_directions[n] = agent_n.direction
            self.population_positions[n] = agent_n.position
            self.population_angular_momenta[n] = agent_n_angular_momentum

        sum_of_directions = np.sum(self.population_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / self.population_size

        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / self.population_size
        

'''env = Environment(100)
pop = Population(population_size=1000, number_of_neighbours=10, environment=env)

dis, ind, tree = pop.find_neighbours()
vectors = pop.calculate_vectors()
#print(f"Distances{dis}")
#print(f"Indices{ind}")
print(f"Initial positions: {pop.population_positions}")
print(f"Vectors: {vectors}")
pop.update_positions()
print(f"Final positions: {pop.population_positions}")'''



