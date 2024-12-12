########################################
##     Tests                          ##
########################################

from environment_class import Environment
from population_class import Population
from parent_class import Parent
import numpy as np
from sklearn.neighbors import KDTree

test_prints = False
env = Environment(10, 3)
pop = Population(10, 5, env)
indices_array = np.arange(10)
select = np.array([1,2,5,7])



target_indices = np.array([np.array([]), np.array([1,3,6,9]), np.array([1,2,3]), np.array([1,2,3,4,5,9]), np.array([7]), 
                           np.array([3,5,6]), np.array([2,3]), np.array([1,9]), np.array([8]), np.array([6,7])], dtype=object)

all_repulsion_vectors = np.empty((pop.population_size, 3))
for index, target in enumerate(target_indices):
    select_indices = np.isin(indices_array, target) # returns a boolean array indicating which elements are in the target array
    where = np.where(select_indices[:,np.newaxis], pop.population_positions, 0)
    pos = pop.population_positions[index]
    #select_positions = np.where(select_indices[:,np.newaxis], pop.population_positions, 0)
    cj_minus_ci = np.where(select_indices[:,np.newaxis], where - pos, 0)
    normals = np.linalg.norm(cj_minus_ci, axis=1)
    within_range = (normals > Parent.ral) & (normals < Parent.rat)
    normalised = np.where(select_indices[:,np.newaxis], cj_minus_ci/(normals[:,np.newaxis] + 1e-6), 0)
    normalised_sum = np.sum(normalised, axis=0)

    repulsion_vector = -normalised_sum
    all_repulsion_vectors[index] = repulsion_vector
    
    if test_prints:
        print(f"Target: \n{target}")
        print(f"Boolean: \n{select_indices}")
        print(f"Relevant boolean: \n{within_range}")
        print(f"Selected positions: \n{where}")
        print(f"Position, ci: \n{pos}")
        print(f"cj_minus_ci: \n{cj_minus_ci}")
        print(f"Normals: \n{normals[:,np.newaxis]}")
        print(f"Normalised: \n{normalised}")
        print(f"Sum: \n{normalised_sum}")
#print(f"All repulsion vectors: \n{all_repulsion_vectors}")

#print(np.tile(indices_array, (len(indices_array),1)))

truth_array = np.full(len(indices_array), True)
target_indices = np.array([np.array([2,3]), np.array([1,3,6,9]), np.array([1,2,3])], dtype=object)
v = np.array([[2,3,0], [3,6,9], [1,2,3]])
#print(np.round(np.array([[0.34, 0.44, 0.54], [0.99, 0.456, 0.33]]), 4))

#print(np.where(truth_array[:,np.newaxis], target_indices, 0))
tiles = np.tile(indices_array, (len(target_indices),1))
targets_positions = np.isin(indices_array, target_indices.reshape(len(target_indices),1)[0])
#print(np.where(tiles, ))

#print(target_indices.reshape(len(target_indices),1)[0])
#print(f"Targets: {targets_positions}")
#print(np.tile(np.array([np.arange(10), np.arange(4)], dtype=object), (10,1)))


all_positions = pop.population_positions
tree = KDTree(all_positions)
rat_neighbours = tree.query_radius(all_positions, Parent.rat, return_distance=True)
ral_neighbours = tree.query_radius(all_positions, Parent.ral)
rr_neighbours = tree.query_radius(all_positions, Parent.rr)

# Find the set difference between these arrays
rz_Nei = rr_neighbours
#alz_Nei = [np.setdiff1d(ral, rr) for ral, rr in zip(ral_neighbours, rr_neighbours)]
#atz_Nei = [np.setdiff1d(rat, ral) for rat, ral in zip(rat_neighbours, ral_neighbours)]



target_indices = rat_neighbours[0]
all_distances = rat_neighbours[1]

def remove_false_zeros(boolean, indices):
    valid_indices = []
    for i, v in enumerate(indices):
        if not ((v == 0) and (not boolean[i])):
            valid_indices.append(v)
    return np.array(valid_indices)

def remove_self_index(index, indices):
    valid_indices = []
    for i in indices:
        if not i == index:
            valid_indices.append(i)
    return np.array(valid_indices)

all_vectors = np.empty((pop.population_size, 3))
for index, distances in enumerate(all_distances):
    zone_condition = (distances > Parent.ral) & (distances < Parent.rat)
    print(zone_condition)
    selected_indices = np.where(zone_condition, target_indices[index], 0)
    print(selected_indices)
    selected_indices = remove_false_zeros(zone_condition, selected_indices)
    print(selected_indices)
    selected_indices = remove_self_index(index, selected_indices)
    print(selected_indices)
    selected_distances = np.where(zone_condition, distances, 0)
    selected_distances = selected_distances[selected_distances != 0]
    #print(selected_distances)
    cjs = pop.population_positions[[c for c in selected_indices]]
    #print(cjs)
    pos = pop.population_positions[index]
    #print(pos)
    cj_minus_ci = cjs - pos
    #print(cj_minus_ci)
    normalised = cj_minus_ci/selected_distances[:,np.newaxis]
    #print(normalised)

    normalised_sum = np.sum(normalised, axis=0)
    #print(normalised_sum)
    all_vectors[index] = -normalised_sum 
#print(all_vectors)

