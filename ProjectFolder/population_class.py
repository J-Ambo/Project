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

    def find_neighbours(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        distances, indices = tree.query(all_positions, k=self.n_neighbours+1)
        return distances, indices, tree
    
    @profile
    def find_neighbours_in_zones(self):
        all_positions = self.population_positions
        tree = KDTree(all_positions)
        rat_neighbours = tree.query_radius(all_positions, Parent.rat, return_distance=True)
        #ral_neighbours = tree.query_radius(all_positions, Parent.ral)
        #rr_neighbours = tree.query_radius(all_positions, Parent.rr)

        # Find the set difference between these arrays
        #alz_Nei = [np.setdiff1d(ral, rr, assume_unique=True) for ral, rr in zip(ral_neighbours, rr_neighbours)]
        #atz_Nei = [np.setdiff1d(rat, ral, assume_unique=True) for rat, ral in zip(rat_neighbours, ral_neighbours)]
        #print(rat_neighbours)
        return rat_neighbours  #rz_Nei, alz_Nei, atz_Nei
    
    '''
    def get_density(self):
        tree = KDTree(self.population_positions)
        density = tree.kernel_density(self.population_positions, h=1.5)
        return density'''

    def calculate_repulsion_vectors(self, neighbours, distances):
        all_repulsion_vectors = np.empty((self.population_size, 3))
        indices_array = np.arange(self.population_size)
        for index, target in enumerate(neighbours):
            select_indices = np.isin(indices_array, target, assume_unique=True) # returns a boolean array indicating which elements are in the target array
            selected_positions = np.where(select_indices[:,np.newaxis], self.population_positions, 0) # returns the positions of the selected indices
            pos = self.population_positions[index]
            cj_minus_c0 = np.where(select_indices[:,np.newaxis], selected_positions - pos, 0)
            normals = np.linalg.norm(cj_minus_c0, axis=1)
            normalised = np.where(select_indices[:,np.newaxis], cj_minus_c0/(normals[:,np.newaxis] + 1e-6), 0)
            normalised_sum = np.sum(normalised, axis=0)

            repulsion_vector = -normalised_sum
            all_repulsion_vectors[index] = repulsion_vector
        return all_repulsion_vectors
    
    def caculate_alignment_vectors(self, neighbours, distances):
        all_alignment_vectors = np.empty((self.population_size, 3))
        indices_array = np.arange(self.population_size)
        for index, target in enumerate(neighbours):
            select_indices = np.isin(indices_array, target)
            selected_directions = np.where(select_indices[:,np.newaxis],self.population_directions, 0)
            normals = np.linalg.norm(selected_directions, axis=1)
            normalised = np.where(select_indices[:,np.newaxis], selected_directions/(normals[:,np.newaxis] + 1e-6), 0)
            normalised_sum = np.sum(normalised, axis=0)

            alignment_vector = normalised_sum
            all_alignment_vectors[index] = alignment_vector
        return all_alignment_vectors

    def calculate_attraction_vectors(self, neighbours, distances):
        all_attraction_vectors = np.empty((self.population_size, 3))
        indices_array = np.arange(self.population_size)
        for index,target in enumerate(neighbours):
            select_indices = np.isin(indices_array, target)
            selected_positions = np.where(select_indices[:,np.newaxis], self.population_positions, 0)
            pos = self.population_positions[index]
            cj_minus_ci = np.where(select_indices[:,np.newaxis], selected_positions - pos, 0)
            normals = np.linalg.norm(cj_minus_ci, axis=1)
            normalised = np.where(select_indices[:,np.newaxis], cj_minus_ci/(normals[:,np.newaxis] + 1e-6), 0)
            normalised_sum = np.sum(normalised, axis=0)

            attraction_vector = normalised_sum
            all_attraction_vectors[index] = attraction_vector
        return all_attraction_vectors
    
    def calculate_wall_vectors(self, environment):
        pass

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

            #print(f"Agent{n}'s position: {agent_n.position}")

            distance_from_origin = np.linalg.norm(agent_n.position)
            distance_from_boundary = environment.radius - distance_from_origin

            agent_n.repulsion_vector = np.zeros(vector_dimension)
            agent_n.alignment_vector = np.zeros(vector_dimension)
            agent_n.attraction_vector = np.zeros(vector_dimension)
            agent_n.wall_vector = np.zeros(vector_dimension)

            if distance_from_origin >= environment.radius:
                agent_n.wall_vector += -agent_n.position
                #print(f'{n} OUT OF BOUNDS!')
                #vectors[n] = np.array([agent_n.repulsion_vector, agent_n.alignment_vector, agent_n.attraction_vector, agent_n.wall_vector])
            elif environment.radius * 0.8 < distance_from_origin <= environment.radius:
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
 
                if agent_n.radius_of_repulsion <= agent_i_distance < agent_n.radius_of_alignment:
                    agent_n.alignment_vector += agent_i.direction

                if agent_n.radius_of_alignment <= agent_i_distance < agent_n.radius_of_attraction:
                    agent_n.attraction_vector += (agent_i.position - agent_n.position)/agent_i_distance
                
            v = np.array([agent_n.repulsion_vector, agent_n.alignment_vector, agent_n.attraction_vector, agent_n.wall_vector])  
            normals = np.linalg.norm(v, axis=1)
            normalised_vectors = v/(normals[:,np.newaxis] + 1e-6)
            vectors[n] = normalised_vectors

        steering_vectors = np.round(vectors, 4)
        return steering_vectors

    def update_positions(self, environment):
        neighbours_distances = self.find_neighbours_in_zones()
        neighbours = neighbours_distances[0]
        distances = neighbours_distances[1]
        print(neighbours_distances)
        print(distances[0])
        within_range =(distances[0] > Parent.ral) & (distances[0] < Parent.rat)
        print(np.where(within_range, True, False))
        repulsion_vectors = self.calculate_repulsion_vectors(neighbours, distances)
        alignment_vectors = self.caculate_alignment_vectors(neighbours, distances)
        attraction_vectors = self.calculate_attraction_vectors(neighbours, distances)
        #sum_of_vectors = np.sum((repulsion_vectors, alignment_vectors, attraction_vectors), axis=0)

        steering_vectors = self.calculate_vectors(environment)
        #print(f"Steering vectors: \n{steering_vectors}")
        sum_of_vectors = np.sum(steering_vectors, axis=1)
        
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
        errors = np.random.normal(0, 0.1, (self.population_size, self.dimension))
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
        






