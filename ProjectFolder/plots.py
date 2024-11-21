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

for _ in range(5):
    r = random.uniform(0, env.radius)
    theta = random.uniform(0, 2*np.pi)

    x = r*np.cos(theta)*0.9
    y = r*np.sin(theta)*0.9
    all_agents.append(Prey(x, y))


#agents_positions = np.array([agent.position for agent in all_agents])

distances, indices, tree = all_agents[0].find_neighbours(all_agents)
focus_point = all_agents[0].position.reshape(1,-1)
nn_rz, nn_alz, nn_atz, i_rz, i_alz, i_atz = all_agents[0].count_neighbours(tree, focus_point)

print("Distances:", distances)
print("Nearest distances:", distances[:, 1:])
print("Indices", indices)
print('')
print(i_rz, i_alz, i_atz)
#for i in range(len(indices)):
    #print(f"i am agent {i}, my second closest neighbour is agent {indices[i,2]} at a distance of {distances[i,2]}")
    #print(indices[i, 1:])
print('')
print(f'{nn_rz}, {nn_alz}, {nn_atz}')

for i in range(len(indices)):
    separation_distances_rz = {n: distances[i, np.where(indices[i] == n)[0][0]] for n in i_rz if n != i}
    print(separation_distances_rz)
