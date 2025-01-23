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
    selfish = True      #are individuals selfish(True) (i.e.only consider escape when threatened, ignore group) or unselfish(False) (i.e. try to maintain group cohesion and escape when threatened)
    def __init__(self, population_size, environment, predator):
        self.population_size = population_size
        self.dimension = environment.dimension
        self.half_perception_angle = Parent.perception_angle / 2
        self._positions_cache = {}
        self._density_cache = {}

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
        #print(self.population_positions)
        self.population_directions = np.array([agent.direction for agent in self.population_array])

        self.all_positions = np.vstack((self.population_positions, predator.position))   #all positions including the predator
        #print(self.all_positions)
        self.all_directions = np.vstack((self.population_directions, predator.direction))
        self.all_agents = np.concatenate([self.population_array, [predator]])
       
        self.population_speeds = np.array([agent.speed for agent in self.population_array])
        self.dimension = environment.dimension
                 
        self.average_school_position = np.zeros(3)
        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter

    def get_tree(self):
        tree = KDTree(self.all_positions)
        return tree
    
    def find_neighbours(self, tree):
        rat_neighbours = tree.query_radius(self.population_positions, Parent.rat, return_distance=True)
        neighbour_indices = rat_neighbours[0]
        neighbour_distances = rat_neighbours[1]
        #print(neighbour_distances[-1], neighbour_indices[-1])
        # Remove hidden indices for each agent
        for index in range(self.population_size):
            neighbour_indices[index], neighbour_distances[index] = self.remove_hidden_indices(index, neighbour_indices[index], neighbour_distances[index])
            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbour_indices[index]])

            #print(np.any(predator_mask))
            if self.selfish:
                if np.any(predator_mask):
                    #if one of the neighbours is a predator add its index only
                    print(neighbour_indices[index])
                    neighbour_indices_copy = neighbour_indices[index][predator_mask].copy()
                    neighbour_distances_copy = neighbour_distances[index][predator_mask].copy()
                    neighbour_indices[index] = neighbour_indices_copy
                    neighbour_distances[index] = neighbour_distances_copy
                    print(neighbour_indices[index])
                
        return neighbour_indices, neighbour_distances

    # Remove neighbours which are within the blind volume. 
    def remove_hidden_indices(self, index, indices, distances):
        self_mask = (distances != 0)
        indices_ex_self = indices[self_mask]
        distances_ex_self = distances[self_mask]
        if indices_ex_self.size == 0:
            return indices_ex_self, np.array([], dtype=bool)    #ensures an early exit of the function for agents with no neighbours
        
        #if not np.any(self_mask):
            #return indices, np.array([False] * len(indices), dtype=bool)    
        cache_key = (index, hash(indices_ex_self.tobytes()), hash(distances_ex_self.tobytes()))
        if cache_key in self._positions_cache:
            return self._positions_cache[cache_key]
        
        neighbour_positions = self.all_positions[indices_ex_self]
        focus_position = self.all_positions[index]

        directions_to_neighbours = (neighbour_positions - focus_position)
        np.divide(directions_to_neighbours, distances_ex_self[:,np.newaxis], out=directions_to_neighbours)

        #print(neighbour_positions, focus_position, neighbour_positions-focus_position, distances,  np.linalg.norm(neighbour_positions - focus_position, axis=1))
        dot_products = np.dot(directions_to_neighbours, self.population_directions[index])
        dot_products = np.round(dot_products, 3)
        #print(dot_products)
        angles_to_neighbours = np.arccos(dot_products)
        #print(index, distances, np.linalg.norm(neighbour_positions - focus_position, axis=1))
        mask = angles_to_neighbours <= self.half_perception_angle
        visible_indices = indices_ex_self[mask]
        visible_distances = distances_ex_self[mask]
        self._positions_cache[cache_key] = (visible_indices, mask)

        return visible_indices, visible_distances

    def calculate_repulsion_vectors(self, neighbours, distances):
        all_repulsion_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            if self.population_size in neighbours[index]:
                continue
            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > 0) & (neighbour_distances < Parent.rr)

            if not np.any(zone_condition):
                continue

            selected_indices = neighbours[index][zone_condition]
            selected_distances = neighbour_distances[zone_condition]

           # print('Repulsion distances', selected_distances)

            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            np.divide(cj_minus_ci, selected_distances[:, np.newaxis], out=cj_minus_ci)

            sum_of_normalised = np.sum(cj_minus_ci, axis=0)
            all_repulsion_vectors[index] = -sum_of_normalised
        return all_repulsion_vectors
    
    def caculate_alignment_vectors(self, neighbours, distances):
        all_alignment_vectors = np.empty((self.population_size, 3))
        for index in range(self.population_size):
            if self.population_size in neighbours[index]:
                continue
            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > Parent.rr) & (neighbour_distances < Parent.ral)

            if not np.any(zone_condition):
                continue

            selected_indices = neighbours[index][zone_condition]
            #selected_distances = neighbour_distances[zone_condition]

           # print('Alignment distances', selected_distances)
            if selected_indices.size == 0:
                continue

            vjs = self.population_directions[selected_indices]
            sum_of_normalised = np.sum(vjs, axis=0)

            all_alignment_vectors[index] = sum_of_normalised
        return all_alignment_vectors

    def calculate_attraction_vectors(self, neighbours, distances):
        all_attraction_vectors = np.empty((self.population_size, 3))
        for index in range(self.population_size):
            if self.population_size in neighbours[index]:
                continue

            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > Parent.ral) & (neighbour_distances < Parent.rat)

            if not np.any(zone_condition):
                continue

            selected_indices = neighbours[index][zone_condition]
            selected_distances = neighbour_distances[zone_condition]

            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            np.divide(cj_minus_ci, selected_distances[:, np.newaxis], out=cj_minus_ci)

            sum_of_normalised = np.sum(cj_minus_ci, axis=0)
            all_attraction_vectors[index] = sum_of_normalised
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

    def caclulate_escape_vectors(self, predator, neighbours, distances):
        all_escape_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            if not self.population_size in neighbours[index]:
                continue
            predator_distance = distances[index]
            zone_condition = (predator_distance > 0) & (predator_distance < Parent.rat)
            if not np.any(zone_condition):
                continue

            cj = predator.position
            pos = self.population_positions[index]

            cj_minus_ci = cj - pos
            print(cj_minus_ci)
            print(predator_distance)
            np.divide(cj_minus_ci, predator_distance, out=cj_minus_ci)

            sum_of_normalised = np.sum(cj_minus_ci, axis=0)
            all_escape_vectors[index] = -sum_of_normalised
        return all_escape_vectors

    def calculate_target_directions(self, tree,  environment, predator):
        neighbours_distances = self.find_neighbours(tree)
        neighbours = neighbours_distances[0]
        distances = neighbours_distances[1]

        repulsion_vectors = self.calculate_repulsion_vectors(neighbours, distances)
        alignment_vectors = self.caculate_alignment_vectors(neighbours, distances)
        attraction_vectors = self.calculate_attraction_vectors(neighbours, distances)
        boundary_vectors = self.calculate_wall_vectors(environment)    
        escape_vector = self.caclulate_escape_vectors(predator, neighbours, distances)

        sum_of_vectors = np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, boundary_vectors, escape_vector), axis=0)
        mask_zero = np.all(sum_of_vectors == 0, axis=1)
        target_directions = np.where(mask_zero[:, np.newaxis], self.population_directions, sum_of_vectors)
        np.divide(target_directions, np.linalg.norm(target_directions, axis=1, keepdims=True),out=target_directions)
        return target_directions
        
    def update_positions(self, tree, environment, predator):
        target_directions = self.calculate_target_directions(tree, environment, predator)
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

    def update_all_positions(self, predator):
        # Use pre-allocated array instead of vstack
        if not hasattr(self, '_all_positions'):
            self._all_positions = np.zeros((self.population_size + 1, 3))
        self._all_positions[:-1] = self.population_positions
        self._all_positions[-1] = predator.position
        self.all_positions = self._all_positions

    def get_densities(self, tree):
        Nn = 10
        distances = tree.query(self.population_positions, k=Nn)[0]
        cache_key = hash(self.population_positions.tobytes())
        if cache_key in self._density_cache:
            return self._density_cache[cache_key]
        
        densities = (Nn-1)/distances[:, -1]
        self._density_cache[cache_key] = densities
        return densities
    
    def remove_outliers(self, tree):
        densities = self.get_densities(tree)

        n_classes = 100
        density_bins = np.linspace(densities.min(), densities.max(), n_classes + 1)
        class_labels = np.digitize(densities, density_bins)

        # Outlier detection
        outlier_threshold = 0.01
        outlier_mask = class_labels <= np.percentile(class_labels, outlier_threshold * 100)
        if not np.any(outlier_mask):
            return self.population_positions, self.population_directions

        return self.population_positions[~outlier_mask], self.population_directions[~outlier_mask]

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






