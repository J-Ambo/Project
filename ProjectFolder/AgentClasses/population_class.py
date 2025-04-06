import numpy as np 
from sklearn.neighbors import KDTree, LocalOutlierFactor
from AgentClasses.parent_class import Parent


from line_profiler import profile

class Population(Parent):
    steering_error = None
    selfish = None
    def __init__(self, population_size, environment, predator):
        self.population_size = population_size
        self.half_perception_angle = Parent.perception_angle / 2
        self.depth = environment.depth

        starting_sphere_radius = 10   #Parent.rat*(population_size/5)**(1/3)
        r = np.random.uniform(0, starting_sphere_radius, self.population_size)
        phi = np.random.uniform(0, 2*np.pi, self.population_size)
        theta = np.random.uniform(0, np.pi, self.population_size)

        z = r * np.cos(theta)    #
        x = r * np.cos(phi)*np.sin(theta)          #
        y = r * np.sin(phi)*np.sin(theta)   #    #

      #  z = self.depth/2   #np.random.uniform(2, self.depth*0.9, self.population_size) 
      #  x = r * np.cos(phi) 
      #  y = r * np.sin(phi)    

        self.population_array = np.array([Parent(x=x[n], y=y[n], z=z[n]) for n in range(self.population_size)], dtype=object)
        self.population_positions = np.concatenate([np.array([agent.position for agent in self.population_array]), [predator.position]])
        self.population_directions = np.concatenate([np.array([agent.direction for agent in self.population_array]), [predator.direction]])
        self.population_speeds = np.array([agent.speed for agent in self.population_array])
        self.population_densities = np.zeros(population_size)

        self.polarisation = 0   # Polarisation order parameter
        self.rotation = 0    # Rotation order parameter
        
       # self.predator_mask = None
        self.escape_coefficient = np.sin(np.pi/2 * Population.selfish)
        self.social_coefficient = np.cos(np.pi/2 * Population.selfish)
        self.is_threatened = False
        self.average_school_position = np.array([0,0,0])
        self.heading_position = np.array([0,0,1000])

    def get_tree(self):
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
    
    def calculate_local_density(self, neighbours_indices):
        density = len(neighbours_indices) / ((4/3)*np.pi*Parent.rat**3)
        #print(len(neighbours_indices), neighbours_indices)
        return density

    def calculate_wall_vectors(self, focus_position, focus_direction, normal_vector):
        dot_product = np.dot(focus_direction, normal_vector)
        if dot_product<0:
            v_w = focus_direction * np.array([1, 1, -1])
            return v_w
        elif dot_product>0:
            return focus_direction

    def calculate_repulsion_vector(self, focus_position, neighbour_positions, neighbour_distances):
        vs = -(neighbour_positions - focus_position) / neighbour_distances[:, np.newaxis]
        v_r = np.sum(vs, axis=0)
        norm = np.linalg.norm(v_r)
        if norm > 0:
            v_r /= norm
        return v_r
       
    def calculate_alignment_vector(self, neighbour_directions):
        v_al = np.sum(neighbour_directions, axis=0)
        norm = np.linalg.norm(v_al)
        if norm > 0:
            v_al /= norm
        return v_al

    def calculate_attraction_vector(self, focus_position, neighbour_positions, neighbour_distances):
        vs = (neighbour_positions - focus_position) / neighbour_distances[:, np.newaxis]
        v_at = np.sum(vs, axis=0)
        norm = np.linalg.norm(v_at)
        if norm > 0:
            v_at /= norm
        return v_at

    def calculate_escape_vector(self, focus_position, predator_distance, predator):
        v = (focus_position - predator.position) / predator_distance
        cross_product = np.cross(v, -predator.direction)
        rotation_axis = cross_product / np.linalg.norm(cross_product)
        ux = rotation_axis[0]
        uy = rotation_axis[1]
        uz = rotation_axis[2]
        c = np.cos(Parent.evasion_angle)      
        s = np.sin(Parent.evasion_angle)
        t = 1 - c
        rotation_matrix = np.array([[c + (ux**2 * t), ux * uy * t - (uz * s), ux * uz * t + (uy * s)],
                                    [uy * ux * t + (uz * s), c + uy**2 * t, uy * uz * t - (ux * s)],
                                    [uz * ux * t - (uy * s), uz * uy * t + (ux * s), c + (uz**2 * t)]])
        v_e = rotation_matrix @ v
        return v_e
    
    def calculate_heading_vector(self, focus_position):
        v_h = self.heading_position - focus_position
        norm = np.linalg.norm(v_h)
        if norm > 0:
            v_h /= norm
        return v_h

    def calculate_all_vectors(self, neighbours, distances, predator):
        self.all_repulsion_vectors = np.zeros((self.population_size, 3))
        self.all_alignment_vectors = np.zeros((self.population_size, 3))
        self.all_attraction_vectors = np.zeros((self.population_size, 3))
        self.all_escape_vectors = np.zeros((self.population_size, 3))
        self.all_wall_vectors = np.zeros((self.population_size, 3))
        self.all_heading_vectors= np.zeros((self.population_size, 3))  #Set

        self.predator_mask = np.zeros(self.population_size, dtype=bool)
        
        self.neighbour_lens = [len(neighbour_array) for neighbour_array in neighbours]

        for index in range(len(self.population_positions[:-1])):
            index_neighbours = neighbours[index]
            index_position = self.population_positions[index]


            if (index_position[-1] < -self.depth/2) | (index_position[-1] > self.depth/2):     # Before looking at neighbours check if the current fish is out of bounds
                normal_vector = np.array([0,0,-index_position[-1]/abs(index_position[-1])])
                self.all_wall_vectors[index] = self.calculate_wall_vectors(index_position, self.population_directions[index], normal_vector)
                continue

            if index_neighbours.size == 0:
                self.population_densities[index] = 0    #np.nan
                self.population_directions[index] = self.average_school_position - self.population_positions[index]
                if (predator.speed != 0) and not (self.is_threatened):
                    self.all_heading_vectors[index] = self.calculate_heading_vector(index_position)
                continue

            if (predator.speed != 0) and not (self.is_threatened):
                self.all_heading_vectors[index] = self.calculate_heading_vector(index_position)

            index_neighbours_distances = distances[index]

            predator_mask = index_neighbours == len(self.population_positions[:-1])  #self.population_size
            self.predator_mask[index] = np.any(predator_mask==True)

            index_social_neighbours_distances = distances[index][~predator_mask]
            index_social_neighbours_positions = self.population_positions[index_neighbours][~predator_mask]
            index_social_neighbours_directions = self.population_directions[index_neighbours][~predator_mask]

            repulsion_zone_mask = (index_social_neighbours_distances > 0) & (index_social_neighbours_distances <= Parent.rr)
            alignment_zone_mask = (index_social_neighbours_distances > Parent.rr) & (index_social_neighbours_distances <= Parent.ral)
            attraction_zone_mask = (index_social_neighbours_distances > Parent.ral) & (index_social_neighbours_distances <= Parent.rat)

            if np.any(index_neighbours == self.population_size):
                predator_distance = index_neighbours_distances[-1]
                self.all_escape_vectors[index] = self.calculate_escape_vector(index_position, predator_distance, predator)
                self.is_threatened = True

            self.population_densities[index] = self.calculate_local_density(index_neighbours)

         ##   print(index_neighbours[alignment_zone_mask], index_social_neighbours_distances[alignment_zone_mask])

         ##   self.all_repulsion_vectors[index] = self.calculate_repulsion_vector(index_position, index_social_neighbours_positions[repulsion_zone_mask], index_social_neighbours_distances[repulsion_zone_mask])
            if index_neighbours[~predator_mask][repulsion_zone_mask].size >= 1:     
                # If the agent has at least one neighbour in its zone of repulsion (i.e. a non zero repulsion vector) then skip the other vector calculations.
                self.all_repulsion_vectors[index] = self.calculate_repulsion_vector(index_position, index_social_neighbours_positions[repulsion_zone_mask], index_social_neighbours_distances[repulsion_zone_mask])
                continue                                            

            self.all_alignment_vectors[index] = self.calculate_alignment_vector(index_social_neighbours_directions[alignment_zone_mask])
            self.all_attraction_vectors[index] = self.calculate_attraction_vector(index_position, index_social_neighbours_positions[attraction_zone_mask], index_social_neighbours_distances[attraction_zone_mask])

    def calculate_target_directions(self, tree, predator):
        neighbours_distances = self.find_neighbours(tree, self.population_positions[:-1])
        neighbours = neighbours_distances[0]
        distances = neighbours_distances[1]

        self.calculate_all_vectors(neighbours, distances, predator)
     #   print(self.population_positions[:-1,-1], self.all_wall_vectors)
        sum_of_social_vectors = self.all_repulsion_vectors + self.all_alignment_vectors + self.all_attraction_vectors + 0.5*self.all_heading_vectors
        sum_of_social_vectors /= (np.linalg.norm(sum_of_social_vectors, axis=1)[:, np.newaxis] + 1e-10)

        all_escape_coefficients = np.ones(self.population_size) * (self.escape_coefficient * self.predator_mask + ~self.predator_mask)
        all_social_coefficients = np.ones(self.population_size) * (self.social_coefficient * self.predator_mask + ~self.predator_mask)
        sum_of_all_vectors = all_social_coefficients[:, np.newaxis] * sum_of_social_vectors + all_escape_coefficients[:, np.newaxis] * self.all_escape_vectors      #np.sum((repulsion_vectors, alignment_vectors, attraction_vectors, 10*escape_vectors), axis=0)

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

     #   self.population_directions[:-1] = np.where(mask[:, np.newaxis], target_directions, maximal_directions)

        boundary_mask = (self.population_positions[:-1,-1] < -self.depth/2) | (self.population_positions[:-1,-1] > self.depth/2)
        self.population_directions[:-1][~boundary_mask] = np.where(mask[:, np.newaxis][~boundary_mask], target_directions[~boundary_mask], maximal_directions[~boundary_mask])

        if np.any(boundary_mask):
            self.population_directions[:-1][boundary_mask] = self.all_wall_vectors[boundary_mask]

        errors = np.random.normal(0, __class__.steering_error, (self.population_size, 3))
        self.population_directions[:-1] += errors
        self.population_directions[:-1] /= np.linalg.norm(self.population_directions[:-1], axis=1)[:, np.newaxis]

    def update_positions(self, tree, environment, predator):
        self.calculate_new_directions(tree, environment, predator)
        self.calculate_average_inlier_position()
        predator.calculate_angles(self.population_positions[:-1], self.population_directions[:-1])  #(self.inlier_positions, self.inlier_directions)

        self.population_positions[:-1] += self.population_speeds[:, np.newaxis] * self.population_directions[:-1] * 0.1

    def remove_outliers(self):
        lof = LocalOutlierFactor(n_neighbors=20, algorithm='kd_tree', contamination='auto')
        outlier_mask = lof.fit_predict(self.population_positions[:-1]) == -1
        
        if not np.any(outlier_mask):
            return self.population_positions[:-1], self.population_directions[:-1]
        
        return self.population_positions[:-1][~outlier_mask], self.population_directions[:-1][~outlier_mask]

    def calculate_average_inlier_position(self):
        inlier_positions, inlier_directions = self.remove_outliers()
        self.average_school_position = np.mean(inlier_positions, axis=0)
        self.inlier_positions = inlier_positions
        self.inlier_directions = inlier_directions
    
    def calculate_order_parameters(self, tree):
        # Calculate average position to each agent

        #densest_position = self.population_positions[np.argmax(self.neighbour_lens)]
        #neighbours_in_shell = tree.query_radius([densest_position], Parent.rat*1.5)
        #self.inlier_positions = self.population_positions[neighbours_in_shell[0]]
        #self.inlier_directions = self.population_directions[neighbours_in_shell[0]]
        #self.average_school_position = np.mean(self.inlier_positions, axis=0)
        #print(school_size)
        
        school_size = len(self.inlier_positions)


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

