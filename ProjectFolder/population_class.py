import numpy as np 
import numpy.ma as ma
import random
from sklearn.neighbors import KDTree, LocalOutlierFactor
from parent_class import Parent


from line_profiler import profile

class Population(Parent):
    steering_error = None
    selfish = 1      #are individuals selfish(1) (i.e.only consider escape when threatened, ignore group) or unselfish(0) (i.e. try to maintain group cohesion when threatened)
    def __init__(self, population_size, environment, predator):
        self.population_size = population_size
        self.half_perception_angle = Parent.perception_angle / 2

        r = np.random.uniform(0, Parent.rat * 1.2, self.population_size)
        phi = np.random.uniform(0, 2*np.pi, self.population_size)
        theta = np.random.uniform(0, np.pi, self.population_size)

        z = r * np.cos(theta)
        x = r * np.cos(phi)*np.sin(theta)
        y = r * np.sin(phi)*np.sin(theta)
        self.population_array = np.array([Parent(x=x[n], y=y[n], z=z[n]) for n in range(self.population_size)], dtype=object)
        self.population_positions = np.concatenate([np.array([agent.position for agent in self.population_array]), [predator.position]])
        self.population_directions = np.concatenate([np.array([agent.direction for agent in self.population_array]), [predator.direction]])
        self.population_speeds = np.array([agent.speed for agent in self.population_array])
            
       # self.all_positions = np.concatenate((self.population_positions, [predator.position]))   #all positions including the predator
       # self.all_directions = np.concatenate((self.population_directions, [predator.direction]))
    
        self.all_repulsion_vectors = None
        self.all_alignment_vectors = None
        self.all_attraction_vectors = None
        self.all_escape_vectors = None

        self.outlier_mask = None
        self.inlier_positions = None
        self.inlier_directions = None
        self.average_school_position = None
        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter
        
        self.predator_mask = None
        self.escape_coefficient = np.sin(np.pi/2 * Population.selfish)
        self.social_coefficient = np.cos(np.pi/2 * Population.selfish)

    def get_tree(self):
       # print('ALL', self.population_positions)
        tree = KDTree(self.population_positions)
        return tree

    # Remove neighbours which are within the blind volume. 
    def remove_hidden_indices(self, index, indices, distances):
        self_mask = (distances != 0)
        indices_ex_self = indices[self_mask]
        distances_ex_self = distances[self_mask]
        if indices_ex_self.size == 0:
            return indices_ex_self, np.array([], dtype=bool)    #ensures an early exit of the function for agents with no neighbours
        
        neighbour_positions = self.population_positions[indices_ex_self]
        focus_position = self.population_positions[index]

        directions_to_neighbours = (neighbour_positions - focus_position)
        np.divide(directions_to_neighbours, distances_ex_self[:,np.newaxis], out=directions_to_neighbours)

        dot_products = np.dot(directions_to_neighbours, self.population_directions[index])
        dot_products = np.round(dot_products, 3)

       # print(dot_products)
        angles_to_neighbours = np.arccos(dot_products)
       # print(angles_to_neighbours)

        mask = angles_to_neighbours <= self.half_perception_angle
        visible_indices = indices_ex_self[mask]
        visible_distances = distances_ex_self[mask]
        return visible_indices, visible_distances
    
    def find_neighbours(self, tree, positions):
        rat_neighbours = tree.query_radius(positions, __class__.rat, return_distance=True)
        neighbour_indices = rat_neighbours[0]
        neighbour_distances = rat_neighbours[1]
        # Remove hidden indices for each agent
        for index in range(len(positions)):
            neighbour_indices[index], neighbour_distances[index] = self.remove_hidden_indices(index, neighbour_indices[index], neighbour_distances[index])
    
        return neighbour_indices, neighbour_distances
    
    def calculate_wall_vectors(self, environment, distances_from_origin):
        all_wall_vectors = np.zeros((self.population_size, 3))
        all_wall_vectors = np.where(distances_from_origin[:, np.newaxis] >= environment.radius, -self.population_positions[:-1], all_wall_vectors)
        '''for index in range(self.population_size):
            position = self.population_positions[index]
            distance_from_origin = np.linalg.norm(position)
            distance_from_boundary = environment.radius - distance_from_origin

            if distance_from_origin >= environment.radius:
                all_wall_vectors[index] = -position
            #else:
                #all_wall_vectors[index] = -position * np.exp(-distance_from_boundary)'''
                
        return all_wall_vectors

    def repulsion_vector(self, focus_position, neighbour_positions, neighbour_distances):
        vs = -(neighbour_positions - focus_position) / neighbour_distances[:, np.newaxis]
        v_r = np.sum(vs, axis=0)
        return v_r

    def alignment_vector(self, neighbour_directions):
        v_al = np.sum(neighbour_directions, axis=0)
        return v_al

    def attraction_vector(self, focus_position, neighbour_positions, neighbour_distances):
        vs = (neighbour_positions - focus_position) / neighbour_distances[:, np.newaxis]
        v_at = np.sum(vs, axis=0)
        return v_at

    def escape_vector(self, focus_position, predator_distance, predator):
        v = (focus_position - predator.position) / predator_distance
        cross_product = np.cross(v, -predator.direction)
        rotation_axis = cross_product / np.linalg.norm(cross_product)
        ux = rotation_axis[0]
        uy = rotation_axis[1]
        uz = rotation_axis[2]
        c = np.cos(Parent.evasion_angle)      
        s = np.sin(Parent.evasion_angle)
        t = 1 - c
        rotation_matrix = np.array([[c + ux**2 * t, ux * uy * t - uz * s, ux * uz * t + uy * s],
                                    [uy * ux * t + uz * s, c + uy**2 * t, uy * uz * t - ux * s],
                                    [uz * ux * t - uy * s, uz * uy * t + ux * s, c + uz**2 * t]])
        v_e = rotation_matrix @ v
        return v_e
    @profile
    def calculate_all_vectors(self, neighbours, distances, predator):
        self.all_repulsion_vectors = np.zeros((self.population_size, 3))
        self.all_alignment_vectors = np.zeros((self.population_size, 3))
        self.all_attraction_vectors = np.zeros((self.population_size, 3))
        self.all_escape_vectors = np.zeros((self.population_size, 3))
        self.predator_mask = np.zeros(self.population_size, dtype=bool)
        self.outlier_mask = np.zeros(self.population_size, dtype=bool)

        for index in range(self.population_size):
            index_neighbours = neighbours[index]
            if index_neighbours.size == 0:
                #print(index_neighbours, self.outlier_mask[0])
                self.outlier_mask[index] = True
                continue
            if index_neighbours.size <= 4:
                self.outlier_mask[index] = True

            index_neighbours_distances = distances[index]
            index_position = self.population_positions[index]

            predator_mask = index_neighbours == self.population_size
            self.predator_mask[index] = np.any(predator_mask==True)
            #if np.any(predator_mask==True):
             #   print(index, predator_mask)
            index_social_neighbours_distances = distances[index][~predator_mask]
            index_social_neighbours_positions = self.population_positions[index_neighbours][~predator_mask]
            index_social_neighbours_directions = self.population_directions[index_neighbours][~predator_mask]

            repulsion_zone_mask = (index_social_neighbours_distances > 0) & (index_social_neighbours_distances < Parent.rr)
            alignment_zone_mask = (index_social_neighbours_distances > Parent.rr) & (index_social_neighbours_distances < Parent.ral)
            attraction_zone_mask = (index_social_neighbours_distances > Parent.ral) & (index_social_neighbours_distances < Parent.rat)

            self.all_repulsion_vectors[index] = self.repulsion_vector(index_position, index_social_neighbours_positions[repulsion_zone_mask], index_social_neighbours_distances[repulsion_zone_mask])
            self.all_alignment_vectors[index] = self.alignment_vector(index_social_neighbours_directions[alignment_zone_mask])
            self.all_attraction_vectors[index] = self.attraction_vector(index_position, index_social_neighbours_positions[attraction_zone_mask], index_social_neighbours_distances[attraction_zone_mask])

            if np.any(index_neighbours == self.population_size):
                predator_distance = index_neighbours_distances[-1]
                self.all_escape_vectors[index] = self.escape_vector(index_position, predator_distance, predator)

    def calculate_target_directions(self, tree, predator):
        neighbours_distances = self.find_neighbours(tree, self.population_positions[:-1])
        neighbours = neighbours_distances[0]
        #print(neighbours)
        distances = neighbours_distances[1]

        self.calculate_all_vectors(neighbours, distances, predator)

        sum_of_social_vectors = self.all_repulsion_vectors + self.all_alignment_vectors + self.all_attraction_vectors
        sum_of_social_vectors /= (np.linalg.norm(sum_of_social_vectors, axis=1)[:, np.newaxis] + 1e-10)

        all_escape_coefficients = np.ones(self.population_size) * (self.escape_coefficient * self.predator_mask + ~self.predator_mask)
        all_social_coefficients = np.ones(self.population_size) * (self.social_coefficient * self.predator_mask + ~self.predator_mask)
        
        sum_of_all_vectors = all_social_coefficients[:, np.newaxis] * sum_of_social_vectors + all_escape_coefficients[:, np.newaxis] * self.all_escape_vectors                     #np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, 10*escape_vectors), axis=0)
        
        mask_zero = np.all(sum_of_all_vectors < 1e-4, axis=1)
        target_directions = np.where(mask_zero[:, np.newaxis], self.population_directions[:-1], sum_of_all_vectors)
        return target_directions
    
    def calculate_new_directions(self, tree, environment, predator):
        target_directions = self.calculate_target_directions(tree, predator)
        
        # Calculate angles to target directions
        dot_products = np.einsum('ij, ij->i', self.population_directions[:-1], target_directions, optimize=True)
        dot_products = np.where(dot_products > 1, 1, dot_products)
        dot_products = np.where(dot_products < -1, -1, dot_products)
        angles_to_target_directions = np.arccos(dot_products)

        # Update directions based on maximal turning angle
        mask = angles_to_target_directions <= Parent.maximal_turning_angle

        cross_products = np.cross(self.population_directions[:-1], target_directions)
        cross_norms = np.linalg.norm(cross_products, axis=1)

        # Handle cases where directions are equal (cross product is zero)
        cross_products = np.where(cross_norms[:, np.newaxis] > 1e-10, cross_products, self.population_directions[:-1])
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
        
        maximal_directions = np.einsum('ijk,ik->ij', rotation_matrices, self.population_directions[:-1])

        distances_from_origin = np.linalg.norm(self.population_positions[:-1], axis=1)
        boundary_mask = distances_from_origin >= environment.radius
        self.population_directions[:-1][~boundary_mask] = np.where(mask[:, np.newaxis][~boundary_mask], target_directions[~boundary_mask], maximal_directions[~boundary_mask])
        if np.any(boundary_mask):
            boundary_vectors = self.calculate_wall_vectors(environment, distances_from_origin)
            boundary_vectors[boundary_mask] /= np.linalg.norm(boundary_vectors[boundary_mask], axis=1)[:, np.newaxis]
            self.population_directions[:-1][boundary_mask] = boundary_vectors[boundary_mask]

        errors = np.random.normal(0, __class__.steering_error, (self.population_size, 3))
        self.population_directions[:-1] += errors

        self.population_directions[:-1] /= np.linalg.norm(self.population_directions[:-1], axis=1)[:, np.newaxis]
        self.population_directions[:-1] = np.round(self.population_directions[:-1], 4)

    def update_positions(self, tree, environment, predator):
        self.calculate_new_directions(tree, environment, predator)
        self.calculate_average_inlier_position()

        self.population_positions[:-1] += self.population_speeds[:, np.newaxis] * self.population_directions[:-1]
        self.population_positions[:-1] = np.round(self.population_positions[:-1], 4)     

    def remove_outliers(self):
        lof = LocalOutlierFactor(n_neighbors=int(0.8*self.population_size), algorithm='kd_tree', contamination='auto')
        outlier_mask = lof.fit_predict(self.population_positions) == -1
       # print(outlier_mask == self.outlier_mask)
        
        if not np.any(outlier_mask):
            return self.population_positions, self.population_directions
        
        return self.population_positions[~outlier_mask], self.population_directions[~outlier_mask]

    def calculate_average_inlier_position(self):
        inlier_positions, inlier_directions = self.remove_outliers()
        self.average_school_position = np.mean(inlier_positions, axis=0)
        self.inlier_positions = inlier_positions
        self.inlier_directions = inlier_directions
    
    def calculate_order_parameters(self):
        # Calculate average position to each agent
        school_size = len(self.inlier_positions)
      #  print(self.population_size-school_size)

        average_position_to_agents = self.inlier_positions - self.average_school_position
        average_position_to_agents /= (np.linalg.norm(average_position_to_agents, axis=1)[:, np.newaxis])
        
        # Calculate order parameters
        sum_of_directions = np.sum(self.inlier_directions, axis=0)
        self.polarisation = np.linalg.norm(sum_of_directions) / school_size
        
        angular_momenta = np.cross(average_position_to_agents, self.inlier_directions)
        self.population_angular_momenta = angular_momenta
        sum_of_angular_momenta = np.sum(self.population_angular_momenta, axis=0)
        self.rotation = np.linalg.norm(sum_of_angular_momenta) / school_size
      #  print(self.polarisation, self.rotation)

