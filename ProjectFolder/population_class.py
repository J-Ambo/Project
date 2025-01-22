import numpy as np 
import random
from sklearn.neighbors import KDTree, LocalOutlierFactor
from parent_class import Parent
from prey_class import Prey
from environment_class import Environment
from predator_class import Predator

from line_profiler import profile

class Population:
    steering_error = 0.15
    def __init__(self, population_size, environment, predator):
        self.population_size = population_size
        self.dimension = environment.dimension

        r = np.random.uniform(0, Parent.rat * 1.2, self.population_size)
        phi = np.random.uniform(0, 2*np.pi, self.population_size)
        if self.dimension == 3:
            theta = np.random.uniform(0, np.pi, self.population_size)
            z = r * np.cos(theta)
        else:
            theta = np.full(self.population_size, np.pi/2)
            z = np.zeros(self.population_size)

        x = r * np.cos(phi)*np.sin(theta)
        y = r * np.sin(phi)*np.sin(theta)
        self.population_array = np.array([Prey(x=x[n], y=y[n], z=z[n], dimensions=self.dimension) for n in range(self.population_size)], dtype=object)
        self.population_positions = np.array([agent.position for agent in self.population_array])
        self.population_directions = np.array([agent.direction for agent in self.population_array])

        self.all_positions = np.vstack((self.population_positions, predator.position))   #all positions including the predator
        self.all_directions = np.vstack((self.population_directions, predator.direction))
        self.all_agents = np.concatenate([self.population_array, [predator]])
       
        self.population_speeds = np.array([agent.speed for agent in self.population_array])
        self.dimension = environment.dimension
                 
        self.average_school_position = np.zeros(3)
        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter

    def get_tree(self):
        tree = KDTree(self.population_positions)
        return tree
    
    def find_neighbours(self, tree):
        rat_neighbours = tree.query_radius(self.population_positions, Parent.rat, return_distance=True)
        neighbour_indices = rat_neighbours[0]
        neighbour_distances = rat_neighbours[1]

        #blind_mask = self.remove_hidden_indices()

        return rat_neighbours#neighbour_indices, neighbour_distances
    
    def get_densities(self, tree):
        distances = tree.query(self.population_positions, k=10)[0]
        densities = np.zeros(self.population_size)
        for n in range(self.population_size):
            nN = len(distances[n])-1
            d = distances[n][-1]
            density = nN/d
            densities[n] = density
        #print(densities)
        return densities
    
    # Remove neighbours which are within the blind volume. 
    def remove_hidden_indices(self, index, indices, distances):
        self_mask = (distances != 0)
        neighbour_positions = self.population_positions[indices][self_mask]
        focus_position = self.population_positions[index]
        #print(self.population_positions[np.array([0,1,2])])
        focus_direction = self.population_directions[index]
        directions_to_neighbours = (neighbour_positions - focus_position) / distances[:,np.newaxis]
        #print(neighbour_positions, focus_position, neighbour_positions-focus_position, distances,  np.linalg.norm(neighbour_positions - focus_position, axis=1))
        dot_products = np.dot(directions_to_neighbours, focus_direction)
        dot_products = np.round(dot_products, 3)
        #print(dot_products)
        angles_to_neighbours = np.arccos(dot_products)
        #print(index, distances, np.linalg.norm(neighbour_positions - focus_position, axis=1))
        mask = angles_to_neighbours <= Parent.perception_angle / 2
        valid_indices = indices[mask]
        return valid_indices, mask

    def calculate_repulsion_vectors(self, neighbours, distances):
        all_repulsion_vectors = np.zeros((self.population_size, 3))
        for index, distances in enumerate(distances):
            zone_condition = (distances > 0) & (distances < Parent.rr)

            selected_indices = neighbours[index][zone_condition]
            selected_distances = distances[zone_condition]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)
            selected_distances = selected_distances[mask]
            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
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

            selected_indices = neighbours[index][zone_condition]
            selected_distances = distances[zone_condition]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)
            if selected_indices.size == 0:
                continue

            vjs = self.population_directions[selected_indices]
            sum_of_normalised = np.sum(vjs, axis=0)
            alignment_vector = sum_of_normalised
            all_alignment_vectors[index] = alignment_vector
        return all_alignment_vectors

    def calculate_attraction_vectors(self, neighbours, distances):
        all_attraction_vectors = np.empty((self.population_size, 3))
        for index, distances in enumerate(distances):
            zone_condition = (distances > Parent.ral) & (distances < Parent.rat)

            selected_indices = neighbours[index][zone_condition]
            selected_distances = distances[zone_condition]

            selected_indices, mask = self.remove_hidden_indices(index, selected_indices, selected_distances)
            selected_distances = selected_distances[mask]
            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
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

    def caclulate_escape_vector(self, predator):
        all_escape_vectors = np.zeros((self.population_size, 3))
        #for index, distances in enumerate(distances):
            
        

    @profile
    def update_positions(self, environment, neighbours, distances):
        
        repulsion_vectors = self.calculate_repulsion_vectors(neighbours, distances)
        alignment_vectors = self.caculate_alignment_vectors(neighbours, distances)
        attraction_vectors = self.calculate_attraction_vectors(neighbours, distances)
        boundary_vectors = self.calculate_wall_vectors(environment)
        sum_of_vectors = np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, boundary_vectors), axis=0)
        
        # Determine target directions
        mask_zero = np.all(sum_of_vectors == 0, axis=1)
        target_directions = np.where(mask_zero[:, np.newaxis], self.population_directions, sum_of_vectors)
        np.divide(target_directions, np.linalg.norm(target_directions, axis=1, keepdims=True),out=target_directions)

        # Calculate angles to target directions
        dot_products = np.einsum('ij, ij->i',self.population_directions, target_directions, optimize=True)
        angles_to_target_directions = np.arccos(np.clip(dot_products, -1.0, 1.0))

        # Update directions based on maximal turning angle
        mask = angles_to_target_directions < Parent.maximal_turning_angle

        cross_products = np.cross(self.population_directions, target_directions)
        cross_norms = np.linalg.norm(cross_products, axis=1)
        # Handle cases where directions are equal (cross product is zero)
        default_axis = np.array([0, 0, 1])
        cross_products = np.where(cross_norms[:, np.newaxis] > 1e-10, cross_products, default_axis)
        cross_norms = np.where(cross_norms > 1e-10, cross_norms, 1.0)

        cross_products /= cross_norms[:, np.newaxis]
        rotation_axes = cross_products
        ux = rotation_axes[:, 0]
        uy = rotation_axes[:, 1]
        uz = rotation_axes[:, 2]
        rotation_matrices = np.zeros((self.population_size, 3, 3))
        c = np.cos(Parent.maximal_turning_angle)      
        s = np.sin(Parent.maximal_turning_angle)
        t = 1 - c
        rotation_matrices = np.array([
            [c + ux**2 * t, ux * uy * t - uz * s, ux * uz * t + uy * s],
            [uy * ux * t + uz * s, c + uy**2 * t, uy * uz * t - ux * s],
            [uz * ux * t - uy * s, uz * uy * t + ux * s, c + uz**2 * t]]).transpose(2, 0, 1)

        maximal_directions = np.einsum('ijk,ik->ij', rotation_matrices, self.population_directions)

        self.population_directions = np.where(mask[:, np.newaxis], target_directions, maximal_directions)
        errors = np.random.normal(0, __class__.steering_error, (self.population_size, 3))
        self.population_directions += errors

        self.population_directions /= np.linalg.norm(self.population_directions, axis=1)[:, np.newaxis]
        if self.dimension == 2:
            self.population_directions[:,-1] = 0

        self.population_directions = np.round(self.population_directions, 4)

        # Update positions
        self.population_positions += self.population_speeds[:, np.newaxis] * self.population_directions
        self.population_positions = np.round(self.population_positions, 4)
        #print(self.population_positions)

    def update_all_positions(self, predator):
        self.all_positions = np.vstack((self.population_positions, predator.position))

    def remove_outliers(self, tree):
        densities = self.get_densities(tree)

        n_classes = 100
        density_bins = np.linspace(densities.min(), densities.max(), n_classes + 1)
        class_labels = np.digitize(densities, density_bins)

        # Outlier detection
        outlier_threshold = 0.01
        outlier_labels = class_labels <= np.percentile(class_labels, outlier_threshold * 100)

        inlier_positions = self.population_positions[~outlier_labels]
        inlier_directions = self.population_directions[~outlier_labels]
        return inlier_positions, inlier_directions

    def calculate_average_inlier_position(self, tree):
        inlier_positions, inlier_directions = self.remove_outliers(tree)
        average_position = np.mean(inlier_positions, axis=0)
        self.average_school_position = average_position
        return inlier_positions, inlier_directions, average_position
    
    def calculate_order_parameters(self, tree):
        # Calculate average position to each agent
        inlier_positions, inlier_directions, average_position = self.calculate_average_inlier_position(tree)
        school_size = len(inlier_positions)
        #print("N outliers:", self.population_size - school_size)

        average_position_to_agents = inlier_positions - average_position
        average_position_to_agents /= (np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis])
        
        # Calculate order parameters
        sum_of_directions = np.sum(inlier_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / school_size

        angular_momenta = np.cross(average_position_to_agents, inlier_directions)
        self.population_angular_momenta = angular_momenta
        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / school_size






