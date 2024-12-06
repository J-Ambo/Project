import numpy as np 
import random
from sklearn.neighbors import KDTree
from parent_class import Parent
from prey_class import Prey
from environment_class import Environment

from line_profiler import profile

class Population:
    def __init__(self, population_size, number_of_neighbours, environment):
        self.population_size = population_size
        self.n_neighbours = number_of_neighbours
        self.dimension = environment.dimension

        r = np.random.uniform(0, Parent.rat, population_size)
        phi = np.random.uniform(0, 2*np.pi, population_size)
        if self.dimension == 3:
            theta = np.random.uniform(0, np.pi, population_size)
            z = r * np.cos(theta)
        else:
            theta = np.full(population_size, np.pi/2)
            z = np.zeros(population_size)

        x = r * np.cos(phi)*np.sin(theta)
        y = r * np.sin(phi)*np.sin(theta)
        self.population_array = np.array([Prey(x=x[n], y=y[n], z=z[n], dimensions=self.dimension) for n in range(population_size)], dtype=object)
        self.population_positions = np.array([agent.position for agent in self.population_array])
        self.population_directions = np.array([agent.direction for agent in self.population_array])
        self.population_speeds = np.array([agent.speed for agent in self.population_array])
        self.dimension = environment.dimension
                 
        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter

    def find_neighbours(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        distances, indices = tree.query(all_positions, k=self.n_neighbours+1)
        return distances, indices, tree
    
    def find_neighbours_in_zones(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        rat_neighbours = tree.query_radius(all_positions, self.population_array[0].radius_of_attraction,)

        ral_neighbours = tree.query_radius(all_positions, self.population_array[0].radius_of_alignment)
        rr_neighbours = tree.query_radius(all_positions, self.population_array[0].radius_of_repulsion)

        rz_Nei = rr_neighbours
        #alz_Nei = np.setdiff1d(ral_neighbours, rr_neighbours)
        #atz_Nei = np.setdiff1d(rat_neighbours, ral_neighbours)
        return rr_neighbours, ral_neighbours, rat_neighbours,# rz_Nei, alz_Nei, atz_Nei
    '''
    def get_density(self):
        tree = KDTree(self.population_positions)
        density = tree.kernel_density(self.population_positions, h=1.5)
        return density'''

    '''def calculate_repulsion_vectors(self):
        for index, target in enumerate(target_indices):
            select_indices = np.isin(indices_array, target) # returns a boolean array indicating which elements are in the target array
            where = np.where(select_indices[:,np.newaxis], self.population_positions, 0)
            pos = self.population_positions[index]
            cj_minus_c0 = np.where(select_indices[:,np.newaxis], where - pos, 0)
            normals = np.linalg.norm(cj_minus_c0, axis=1)
            normalised = np.where(select_indices[:,np.newaxis], cj_minus_c0/(normals[:,np.newaxis] + 1e-6), cj_minus_c0)
            normalised_sum = np.sum(normalised, axis=0)

            repulsion_vector = -normalised_sum'''
        


    def calculate_vectors(self, environment):
        distances, indices, tree = self.find_neighbours()
        vector_dimension = environment.dimension
        vectors = np.zeros((self.population_size, 4, vector_dimension))
        
        for n in range(self.population_size):
            nearest_neighbour_distances = distances[n,1:]
            nearest_neighbour_indices = indices[n,1:]
            agent_n = self.population_array[n]
            agent_n.position = self.population_positions[n]
            agent_n.direction = self.population_directions[n]

            distance_from_origin = np.linalg.norm(agent_n.position)
            distance_from_boundary = environment.radius - distance_from_origin

            agent_n.repulsion_vector = np.zeros(vector_dimension)
            agent_n.alignment_vector = np.zeros(vector_dimension)
            agent_n.attraction_vector = np.zeros(vector_dimension)
            agent_n.wall_vector = np.zeros(vector_dimension)

            if distance_from_origin >= environment.radius:
                agent_n.wall_vector += -agent_n.position
                continue
            elif environment.radius * 0.9 < distance_from_origin <= environment.radius:
                agent_n.wall_vector += -agent_n.position * np.exp(-distance_from_boundary)

            for i in range(self.n_neighbours):
                agent_i_index = nearest_neighbour_indices[i]
                agent_i = self.population_array[agent_i_index]
                agent_i_distance = max(nearest_neighbour_distances[i], 0.01)  ######## Attention needed here ######
                agent_i.position = self.population_positions[agent_i_index]
                agent_i.direction = self.population_directions[agent_i_index]

                direction_to_agent_i = (agent_i.position - agent_n.position) / agent_i_distance
                dot_product = np.dot(agent_n.direction, direction_to_agent_i)
                angle_to_agent_i = np.arccos(max(min(dot_product, 1.0), -1.0))

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
        steering_vectors = np.round(vectors, 4)
        return steering_vectors

    @profile
    def update_positions(self, environment):
        steering_vectors = self.calculate_vectors(environment)
        sum_of_vectors = np.sum(steering_vectors, axis=1)
        # Calculate average position to each agent
        sum_of_vectors = np.sum(steering_vectors, axis=1)
        # Calculate average position to each agent
        average_position = np.mean(self.population_positions, axis=0)
        average_position_to_agents = self.population_positions - average_position
        average_position_to_agents /= np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis]
        
        # Determine target directions
        target_directions = np.where(np.all(sum_of_vectors == 0, axis=1)[:, np.newaxis], self.population_directions, sum_of_vectors)
        target_directions /= np.linalg.norm(target_directions, axis=1)[:, np.newaxis]
        target_directions = np.round(target_directions, 4)
        target_directions = np.round(target_directions, 4)

        # Calculate angles to target directions
        dot_products = np.einsum('ij, ij->i',self.population_directions, target_directions)
        angles_to_target_directions = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Update directions based on maximal turning angle
        mask = angles_to_target_directions < self.population_array[0].maximal_turning_angle

        comparison = np.all(self.population_directions == target_directions, axis=1)
        cross_products = np.where(comparison[:,np.newaxis], self.population_directions, np.cross(self.population_directions, target_directions))
        cross_norms = np.linalg.norm(cross_products, axis=1)
    
        cross_products /= cross_norms[:, np.newaxis]
        sin_angles = cross_norms
        cos_angles = dot_products
        rotation_axes = cross_products
        rotation_matrices = np.zeros((self.population_size, 3, 3))
        c = np.cos(self.population_array[0].maximal_turning_angle * 0.1)      #0.1s is the time step
        s = np.sin(self.population_array[0].maximal_turning_angle * 0.1)
        for i in range(self.population_size):
            ux, uy, uz = rotation_axes[i]
            rotation_matrices[i] = np.array([
                [c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s],
                [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c)]])
            
        maximal_directions = np.einsum('ijk,ik->ij', rotation_matrices, self.population_directions)

        self.population_directions = np.where(mask[:, np.newaxis], target_directions, maximal_directions)
        errors = np.random.normal(0, 0.4, (self.population_size, self.dimension))
        self.population_directions += errors

        self.population_directions /= np.linalg.norm(self.population_directions, axis=1)[:, np.newaxis]
        self.population_directions = np.round(self.population_directions, 4)

        # Update positions
        self.population_positions += self.population_speeds[:, np.newaxis] * self.population_directions
        self.population_positions = np.round(self.population_positions, 4)

        # Calculate order parameters
        angular_momenta = np.cross(average_position_to_agents, self.population_directions)
        self.population_angular_momenta = angular_momenta
        sum_of_directions = np.sum(self.population_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / self.population_size
        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / self.population_size
        






