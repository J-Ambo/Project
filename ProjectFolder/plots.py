import numpy as np
import random
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from prey_class import Prey
from environment_class import Environment

'''def calculate_distances(agents_positions):
    # Assuming agents_positions is a 2D array where each row corresponds to an agent's position
    diff = agents_positions[:, np.newaxis, :] - agents_positions[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    distances = np.maximum(distances, 0.01)  # Apply the minimum distance constraint
    return distances

# Example usage
agents_positions = np.array([[1.0, 1.0], [4.0, 5.0], [-1, 1], [2, 1]])
distances = calculate_distances(agents_positions)
print(distances)
# Return the values where i > j
values_i_greater_j = distances[np.triu_indices(distances.shape[0], k=1)]
#print(values_i_greater_j)'''

env = Environment(10)
all_agents = []

for _ in range(3):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r*np.cos(theta)*0.9
    y = r*np.sin(theta)*0.9
    all_agents.append(Prey(x, y))

agents_positions = np.array([agent.position for agent in all_agents])

distances, indices, indices_repulsion_radius_neighbours = all_agents[0].calcualte_neighbours(all_agents)

print("Distances:", distances)
print("Nearest distances:", distances[:, 1:])
print("Indices:", indices_repulsion_radius_neighbours)




#query_point = agents_positions[1].reshape(1, -1)
#indices = tree.query_radius(query_point, r=6.0)

#print("Indices within radius:", indices)
#print("Nearest Distances:", )
#print("Nearest Indices:", nearest_indices)

