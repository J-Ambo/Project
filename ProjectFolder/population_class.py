import numpy as np 
import numpy.ma as ma
import random
from sklearn.neighbors import KDTree, LocalOutlierFactor
from parent_class import Parent
from prey_class import Prey
from environment_class import Environment
from predator_class import Predator

from line_profiler import profile

class Population:
    steering_error = 0.15
    selfish = False      #are individuals selfish(True) (i.e.only consider escape when threatened, ignore group) or unselfish(False) (i.e. try to maintain group cohesion when threatened as well as escape)
    def __init__(self, population_size, environment, predator):
        self.population_size = population_size
        self.dimension = environment.dimension
        self.half_perception_angle = Parent.perception_angle / 2
        self._positions_cache = {}
        self._density_cache = {}

        r = np.random.uniform(0, Parent.rat * 1.2, self.population_size)
        phi = np.random.uniform(0, 2*np.pi, self.population_size)
        theta = np.random.uniform(0, np.pi, self.population_size)

        z = r * np.cos(theta)
        x = r * np.cos(phi)*np.sin(theta)
        y = r * np.sin(phi)*np.sin(theta)
        self.population_array = np.array([Prey(x=x[n], y=y[n], z=z[n], dimensions=self.dimension) for n in range(self.population_size)], dtype=object)
        self.population_positions = np.array([agent.position for agent in self.population_array])
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
    
    def find_neighbours(self, tree):
        rat_neighbours = tree.query_radius(self.population_positions, Parent.rat, return_distance=True)
        neighbour_indices = rat_neighbours[0]
        neighbour_distances = rat_neighbours[1]
        # Remove hidden indices for each agent
        for index in range(self.population_size):
            neighbour_indices[index], neighbour_distances[index] = self.remove_hidden_indices(index, neighbour_indices[index], neighbour_distances[index])
            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbour_indices[index]])

            #print(np.any(predator_mask))
            if self.selfish:
                if np.any(predator_mask):
                    #if one of the neighbours is a predator add its index only
                    neighbour_indices_copy = neighbour_indices[index][predator_mask].copy()
                    neighbour_distances_copy = neighbour_distances[index][predator_mask].copy()
                    neighbour_indices[index] = neighbour_indices_copy
                    neighbour_distances[index] = neighbour_distances_copy
                
        return neighbour_indices, neighbour_distances
    
    def calculate_repulsion_vectors(self, neighbours, distances):
        all_repulsion_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > 0) & (neighbour_distances < Parent.rr)
            if not np.any(zone_condition):
                continue

            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbours[index][zone_condition]])
            selected_indices = neighbours[index][zone_condition][~predator_mask]
            selected_distances = neighbour_distances[zone_condition][~predator_mask]

            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            np.divide(cj_minus_ci, selected_distances[:, np.newaxis], out=cj_minus_ci)

            sum_of_normalised = np.sum(cj_minus_ci, axis=0)
            all_repulsion_vectors[index] = -sum_of_normalised
            #print(index, neighbour_distances[zone_condition], predator_mask, selected_distances, cjs-pos, cj_minus_ci)
        return all_repulsion_vectors
    
    def calculate_alignment_vectors(self, neighbours, distances):
        all_alignment_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > Parent.rr) & (neighbour_distances < Parent.ral)
            if not np.any(zone_condition):
                continue
            
            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbours[index][zone_condition]])
            selected_indices = neighbours[index][zone_condition][~predator_mask]
    
            if selected_indices.size == 0:
                continue

            vjs = self.population_directions[selected_indices]
            sum_of_normalised = np.sum(vjs, axis=0)
            all_alignment_vectors[index] = sum_of_normalised
        return all_alignment_vectors

    def calculate_attraction_vectors(self, neighbours, distances):
        all_attraction_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            neighbour_distances = distances[index]
            zone_condition = (neighbour_distances > Parent.ral) & (neighbour_distances < Parent.rat)
            if not np.any(zone_condition):
                continue

            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbours[index][zone_condition]])
            selected_indices = neighbours[index][zone_condition][~predator_mask]
            selected_distances = neighbour_distances[zone_condition][~predator_mask]

            if selected_indices.size == 0:
                continue

            cjs = self.population_positions[selected_indices]
            pos = self.population_positions[index]
            cj_minus_ci = cjs - pos
            np.divide(cj_minus_ci, selected_distances[:, np.newaxis], out=cj_minus_ci)

            sum_of_normalised = np.sum(cj_minus_ci, axis=0)
            all_attraction_vectors[index] = sum_of_normalised
        return all_attraction_vectors
    
    def calculate_wall_vectors(self, environment, distances_from_origin):
        all_wall_vectors = np.zeros((self.population_size, 3))
        all_wall_vectors = np.where(distances_from_origin[:, np.newaxis] >= environment.radius, -self.population_positions, all_wall_vectors)
        '''for index in range(self.population_size):
            position = self.population_positions[index]
            distance_from_origin = np.linalg.norm(position)
            distance_from_boundary = environment.radius - distance_from_origin

            if distance_from_origin >= environment.radius:
                all_wall_vectors[index] = -position
            #else:
                #all_wall_vectors[index] = -position * np.exp(-distance_from_boundary)'''
                
        return all_wall_vectors

    def caclulate_escape_vectors(self, predator, neighbours, distances):
        all_escape_vectors = np.zeros((self.population_size, 3))
        for index in range(self.population_size):
            predator_mask = np.array([isinstance(self.all_agents[neighbour], Predator) for neighbour in neighbours[index]])
            if not np.any(predator_mask):
                continue
    
            predator_distance = distances[index][predator_mask]
            cj = predator.position
            pos = self.population_positions[index]

            cj_minus_ci = cj - pos
            #print(index, cj,  cj_minus_ci, predator_distance)
            #np.divide(cj_minus_ci, predator_distance, out=cj_minus_ci)
            all_escape_vectors[index] = -cj_minus_ci
        return all_escape_vectors

    def calculate_target_directions(self, tree, predator):
        neighbours_distances = self.find_neighbours(tree)
        neighbours = neighbours_distances[0]
        distances = neighbours_distances[1]

        repulsion_vectors = self.calculate_repulsion_vectors(neighbours, distances)
        alignment_vectors = self.calculate_alignment_vectors(neighbours, distances)
        attraction_vectors = self.calculate_attraction_vectors(neighbours, distances)  
        escape_vectors = self.caclulate_escape_vectors(predator, neighbours, distances)
        #print(repulsion_vectors,)
        sum_of_vectors = np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, 10*escape_vectors), axis=0)
        mask_zero = np.all(sum_of_vectors < 1e-5, axis=1)
        target_directions = np.where(mask_zero[:, np.newaxis], self.population_directions, sum_of_vectors)
        np.divide(target_directions, np.linalg.norm(target_directions, axis=1, keepdims=True),out=target_directions)
        #print(target_directions)
        return target_directions
    
    def calculate_new_directions(self, tree, environment, predator):
        target_directions = self.calculate_target_directions(tree, predator)
        
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

        distances_from_origin = np.linalg.norm(self.population_positions, axis=1)
        boundary_mask = distances_from_origin >= environment.radius
        self.population_directions[~boundary_mask] = np.where(mask[:, np.newaxis][~boundary_mask], target_directions[~boundary_mask], maximal_directions[~boundary_mask])
        if np.any(boundary_mask):
            boundary_vectors = self.calculate_wall_vectors(environment, distances_from_origin)
            boundary_vectors[boundary_mask] /= np.linalg.norm(boundary_vectors[boundary_mask], axis=1)[:, np.newaxis]
            self.population_directions[boundary_mask] = boundary_vectors[boundary_mask]

        errors = np.random.normal(0, __class__.steering_error, (self.population_size, 3))
        self.population_directions += errors

        self.population_directions /= np.linalg.norm(self.population_directions, axis=1)[:, np.newaxis]
        self.population_directions = np.round(self.population_directions, 4)

    def update_positions(self, tree, environment, predator):
        self.calculate_new_directions(tree, environment, predator)
        self.population_positions += self.population_speeds[:, np.newaxis] * self.population_directions
        self.population_positions = np.round(self.population_positions, 4)

    def update_all_positions(self, predator):
        # Use pre-allocated array instead of vstack
        if not hasattr(self, '_all_positions'):
            self._all_positions = np.zeros((self.population_size + 1, 3))
        self._all_positions[:-1] = self.population_positions
        self._all_positions[-1] = predator.position
        self.all_positions = self._all_positions
    
    def remove_outliers(self):
        lof = LocalOutlierFactor(n_neighbors=20, algorithm='kd_tree', contamination='auto')
        outlier_mask = lof.fit_predict(self.population_positions) == -1
        
        if not np.any(outlier_mask):
            return self.population_positions, self.population_directions
        
        return self.population_positions[~outlier_mask], self.population_directions[~outlier_mask]

    def calculate_average_inlier_position(self):
        inlier_positions, inlier_directions = self.remove_outliers()
        self.average_school_position = np.mean(inlier_positions, axis=0)
        return inlier_positions, inlier_directions
    
    def calculate_order_parameters(self):
        # Calculate average position to each agent
        inlier_positions, inlier_directions = self.calculate_average_inlier_position()
        school_size = len(inlier_positions)
        #print("N outliers:", self.population_size - school_size)

        average_position_to_agents = inlier_positions - self.average_school_position
        average_position_to_agents /= (np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis])
        
        # Calculate order parameters
        sum_of_directions = np.sum(inlier_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / school_size

        angular_momenta = np.cross(average_position_to_agents, inlier_directions)
        self.population_angular_momenta = angular_momenta
        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / school_size






