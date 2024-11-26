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
            self.population_directions[n] = agent.direction
            self.population_speeds[n] = agent.speed

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
            agent_n.position = self.population_positions[n]
            agent_n.direction = self.population_directions[n]

            distance_from_origin = np.linalg.norm(agent_n.position)
            distance_from_boundary = environment.radius - distance_from_origin

            agent_n.repulsion_vector = np.zeros(2)
            agent_n.alignment_vector = np.zeros(2)
            agent_n.attraction_vector = np.zeros(2)
            agent_n.wall_vector = np.zeros(2)

            if distance_from_origin >= environment.radius:
                agent_n.wall_vector += -agent_n.position
                continue
            elif environment.radius * 0.9 < distance_from_origin <= environment.radius:
                agent_n.wall_vector += -agent_n.position * np.exp(-distance_from_boundary)

            for i in range(self.n_neighbours):
                agent_i_index = nearest_neighbour_indices[i]
                agent_i = self.population_array[agent_i_index]
                agent_i_distance = nearest_neighbour_distances[i]

                direction_to_agent_i = (agent_i.position - agent_n.position) / agent_i_distance
                angle_to_agent_i = np.arccos(np.clip(np.dot(agent_n.direction, direction_to_agent_i), -1.0, 1.0))

                if angle_to_agent_i > agent_n.perception_angle/2:
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

        # Calculate average position to each agent
        average_position_to_agents = self.population_positions - average_position
        average_position_to_agents /= np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis]
        
        # Sum of steering vectors
        sum_of_vectors = np.sum(steering_vectors, axis=1)
        
        # Determine target directions
        target_directions = np.where(np.all(sum_of_vectors == 0, axis=1)[:, np.newaxis], self.population_directions, sum_of_vectors)
        target_directions /= np.linalg.norm(target_directions, axis=1)[:, np.newaxis]

        # Calculate angles to target directions
        dot_products = np.einsum('ij,ij->i', self.population_directions, target_directions)
        angles_to_target_directions = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Update directions based on maximal turning angle
        maximal_angles = np.full(self.population_size, self.population_array[0].maximal_turning_angle)
        maximal_rotation_matrices = np.zeros((self.population_size, 2, 2))
        c = np.cos(maximal_angles)
        s = np.sin(maximal_angles)
        maximal_rotation_matrices[:, 0, 0] = c
        maximal_rotation_matrices[:, 0, 1] = -s
        maximal_rotation_matrices[:, 1, 0] = s
        maximal_rotation_matrices[:, 1, 1] = c

        mask = angles_to_target_directions < self.population_array[0].maximal_turning_angle
        z_cross_component = self.population_directions[:, 0] * target_directions[:, 1] - self.population_directions[:, 1] * target_directions[:, 0]
        z_cross_component = z_cross_component[:, np.newaxis, np.newaxis]
        maximal_directions = np.where(z_cross_component > 0, 
                          np.matmul(maximal_rotation_matrices, self.population_directions[:, :, np.newaxis]),
                          np.matmul(np.transpose(maximal_rotation_matrices, axes=(0, 2, 1)), self.population_directions[:, :, np.newaxis]))
        maximal_directions = maximal_directions[:,:,0]

        self.population_directions = np.where(mask[:, np.newaxis], target_directions, maximal_directions)
        
        # Apply random rotation and normalize
        random_angles = np.random.normal(0, 0.2, self.population_size)
        random_rotation_matrices = np.zeros((self.population_size, 2, 2))
        c = np.cos(random_angles)
        s = np.sin(random_angles)
        random_rotation_matrices[:, 0, 0] = c
        random_rotation_matrices[:, 0, 1] = -s
        random_rotation_matrices[:, 1, 0] = s
        random_rotation_matrices[:, 1, 1] = c
        self.population_directions = np.matmul(random_rotation_matrices, self.population_directions[:, :, np.newaxis])
        self.population_directions = self.population_directions[:, :, 0]

        # Update positions
        self.population_positions += self.population_speeds[:, np.newaxis] * self.population_directions

        # Calculate angular momentum
        angular_momenta = np.cross(average_position_to_agents, self.population_directions)
        self.population_angular_momenta = angular_momenta

        # Calculate polarisation and rotation
        sum_of_directions = np.sum(self.population_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / self.population_size

        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / self.population_size
        






