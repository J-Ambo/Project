import numpy as np 
import random
from sklearn.neighbors import KDTree
from parent_class import Parent
from prey_class import Prey
from environment_class import Environment

from line_profiler import profile

class Population:
    def __init__(self, population_size, environment):
        self.population_size = population_size
        self.dimension = environment.dimension

        r = np.random.uniform(0, Parent.rat * 1.2, population_size)
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

    '''def find_neighbours(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        distances, indices = tree.query(all_positions, k=self.n_neighbours+1)
        return distances, indices, tree'''
    
    @profile
    def find_neighbours_in_zones(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        rat_neighbours = tree.query_radius(all_positions, Parent.rat, return_distance=True)
        return rat_neighbours
    
    '''
    def get_density(self):
        tree = KDTree(self.population_positions)
        density = tree.kernel_density(self.population_positions, h=1.5)
        return density'''
    
    # Remove indices=0 if their corresponding boolean value is False, otherwise(if True) keep. 
    def remove_false_zeros(self, boolean, indices):
        boolean = np.asarray(boolean)
        indices = np.asarray(indices)
        mask = ~((indices == 0) & (~boolean))
        valid_indices = indices[mask]
        return valid_indices
    
    # Remove neighbours which are within the blind volume. 
    def remove_hidden_indices(self, index, indices, distances):
        neighbour_positions = self.population_positions[[c for c in indices]]
        focus_position = self.population_positions[index]
        focus_direction = self.population_directions[index]
        directions_to_neighbours = (neighbour_positions - focus_position) / distances[:,np.newaxis]
        dot_products = np.dot(directions_to_neighbours, focus_direction)
        dot_products = np.round(dot_products, 3)
        #print(f"Greater than one: {np.any(dot_products>1)}")
        #print(f"Less than minus one: {np.any(dot_products<-1)}")
        angles_to_neighbours = np.arccos(dot_products)
        mask = angles_to_neighbours <= Parent.perception_angle / 2
        valid_indices = indices[mask]
        return valid_indices, mask

    def calculate_repulsion_vectors(self, neighbours, distances):
        all_repulsion_vectors = np.empty((self.population_size, 3))
        for index, distances in enumerate(distances):
            zone_condition = (distances > 0) & (distances < Parent.rr)

            selected_indices = np.where(zone_condition, neighbours[index], 0)
            selected_distances = np.where(zone_condition, distances, 0)

            selected_indices = self.remove_false_zeros(zone_condition, selected_indices)
            selected_distances = selected_distances[selected_distances != 0]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)
            selected_distances = selected_distances[mask]
            
            cjs = self.population_positions[[c for c in selected_indices]]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            normalised = cj_minus_ci/selected_distances[:,np.newaxis]
            sum_of_normalised = np.sum(normalised, axis=0)

            repulsion_vector = -sum_of_normalised
            all_repulsion_vectors[index] = repulsion_vector
        return all_repulsion_vectors
    
    def caculate_alignment_vectors(self, neighbours, distances):
        all_alignment_vectors = np.empty((self.population_size, 3))
        for index, distances in enumerate(distances):
            zone_condition = (distances > Parent.rr) & (distances < Parent.ral)

            selected_indices = np.where(zone_condition, neighbours[index], 0)
            selected_distances = np.where(zone_condition, distances, 0)

            selected_indices = self.remove_false_zeros(zone_condition, selected_indices)
            selected_distances = selected_distances[selected_distances != 0]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)

            vjs = self.population_directions[[v for v in selected_indices]]
            sum_of_normalised = np.sum(vjs, axis=0)
            alignment_vector = sum_of_normalised
            all_alignment_vectors[index] = alignment_vector
        return all_alignment_vectors

    def calculate_attraction_vectors(self, neighbours, distances):
        all_attraction_vectors = np.empty((self.population_size, 3))
        for index, distances in enumerate(distances):
            zone_condition = (distances > Parent.ral) & (distances < Parent.rat)

            selected_indices = np.where(zone_condition, neighbours[index], 0)
            selected_distances = np.where(zone_condition, distances, 0)

            selected_indices = self.remove_false_zeros(zone_condition, selected_indices)
            selected_distances = selected_distances[selected_distances != 0]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)
            selected_distances = selected_distances[mask]

            cjs = self.population_positions[[c for c in selected_indices]]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            normalised = cj_minus_ci/selected_distances[:,np.newaxis]
            sum_of_normalised = np.sum(normalised, axis=0)

            attraction_vector = sum_of_normalised
            all_attraction_vectors[index] = attraction_vector
        return all_attraction_vectors
    
    def calculate_wall_vectors(self, environment):
        all_wall_vectors = np.zeros((self.population_size, 3))
        for index, position in enumerate(self.population_positions):
            distance_from_origin = np.linalg.norm(position)
            distance_from_boundary = environment.radius - distance_from_origin

            if distance_from_origin >= environment.radius:
                all_wall_vectors[index] = -position
                
            elif environment.radius * 0.8 < distance_from_origin <= environment.radius:
                all_wall_vectors[index] = -position * np.exp(-distance_from_boundary)
        return all_wall_vectors

    def update_positions(self, environment):
        neighbours_distances = self.find_neighbours_in_zones()
        neighbours = neighbours_distances[0]
        distances = neighbours_distances[1]
       
        repulsion_vectors = self.calculate_repulsion_vectors(neighbours, distances)
        alignment_vectors = self.caculate_alignment_vectors(neighbours, distances)
        attraction_vectors = self.calculate_attraction_vectors(neighbours, distances)
        boundary_vectors = self.calculate_wall_vectors(environment)
        sum_of_vectors = np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, boundary_vectors), axis=0)
       
        # Calculate average position to each agent
        average_position = np.mean(self.population_positions, axis=0)
        average_position_to_agents = self.population_positions - average_position
        average_position_to_agents /= np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis]
        
        # Determine target directions
        target_directions = np.where(np.all(sum_of_vectors == 0, axis=1)[:, np.newaxis], self.population_directions, sum_of_vectors)
        target_directions /= np.linalg.norm(target_directions, axis=1)[:, np.newaxis]
        target_directions = np.round(target_directions, 4)

        # Calculate angles to target directions
        dot_products = np.einsum('ij, ij->i',self.population_directions, target_directions)
        angles_to_target_directions = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Update directions based on maximal turning angle
        mask = angles_to_target_directions < Parent.maximal_turning_angle

        comparison = np.all(self.population_directions == target_directions, axis=1)
        cross_products = np.where(comparison[:,np.newaxis], self.population_directions, np.cross(self.population_directions, target_directions))
        cross_norms = np.linalg.norm(cross_products, axis=1)
    
        cross_products /= cross_norms[:, np.newaxis]
        rotation_axes = cross_products
        rotation_matrices = np.zeros((self.population_size, 3, 3))
        c = np.cos(Parent.maximal_turning_angle)      
        s = np.sin(Parent.maximal_turning_angle)
        for i in range(self.population_size):
            ux, uy, uz = rotation_axes[i]
            rotation_matrices[i] = np.array([
                [c + ux**2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                [uy * ux * (1 - c) + uz * s, c + uy**2 * (1 - c), uy * uz * (1 - c) - ux * s],
                [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz**2 * (1 - c)]])
            
        maximal_directions = np.einsum('ijk,ik->ij', rotation_matrices, self.population_directions)

        self.population_directions = np.where(mask[:, np.newaxis], target_directions, maximal_directions)
        errors = np.random.normal(0, 0.25, (self.population_size, self.dimension))
        self.population_directions += errors

        self.population_directions /= np.linalg.norm(self.population_directions, axis=1)[:, np.newaxis]
        self.population_directions = np.round(self.population_directions, 4)

        # Update positions
        self.population_positions += self.population_speeds[:, np.newaxis] * self.population_directions
        self.population_positions = np.round(self.population_positions, 4)

        # Calculate order parameters
        sum_of_directions = np.sum(self.population_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / self.population_size

        angular_momenta = np.cross(average_position_to_agents, self.population_directions)
        self.population_angular_momenta = angular_momenta
        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / self.population_size
        






