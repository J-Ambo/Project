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
pop = Population(10, env)
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
Neighbours = tree.query(all_positions, k=pop.population_size)
distances = np.asarray(Neighbours[0])
neighbours = np.asarray(Neighbours[1])
print(f"distances: \n{distances}")
print(f"neighbours: \n{neighbours}")
rat_neighbours = tree.query_radius(all_positions, Parent.rat, return_distance=True)
ral_neighbours = tree.query_radius(all_positions, Parent.ral)
rr_neighbours = tree.query_radius(all_positions, Parent.rr)

# Find the set difference between these arrays
rz_Nei = rr_neighbours
#alz_Nei = [np.setdiff1d(ral, rr) for ral, rr in zip(ral_neighbours, rr_neighbours)]
#atz_Nei = [np.setdiff1d(rat, ral) for rat, ral in zip(rat_neighbours, ral_neighbours)]



target_indices = rat_neighbours[0]
all_distances = rat_neighbours[1]

'''def remove_false_zeros(boolean, indices):
    valid_indices = []
    for i, v in enumerate(indices):
        if not ((v == 0) and (not boolean[i])):
            valid_indices.append(v)
    return np.array(valid_indices)'''

def remove_self_index(index, indices):
    valid_indices = []
    for i in indices:
        if not i == index:
            valid_indices.append(i)
    return np.array(valid_indices)

def remove_false_zeros(boolean, indices):
        boolean = np.asarray(boolean)
        indices = np.asarray(indices)
        mask = ~((indices == 0) & (~boolean))
        valid_indices = indices[mask]
        return valid_indices
    
# Remove neighbours which are within the blind volume. 
def remove_hidden_indices(index, indices, distances):
    neighbour_positions = pop.population_positions[indices]
    #print(f"neighbour pos:\n{neighbour_positions}")
    focus_position = pop.population_positions[index]
    focus_direction = pop.population_directions[index]
    directions_to_neighbours = (neighbour_positions - focus_position) / distances[:,np.newaxis]
    dot_products = np.dot(directions_to_neighbours, focus_direction)
    angles_to_neighbours = np.arccos(dot_products)
    mask = angles_to_neighbours <= Parent.perception_angle / 2
    valid_indices = indices[mask]
    return valid_indices, mask

all_vectors = np.empty((pop.population_size, 3))
#print(f"All conditions: {[(distances > Parent.ral) & (distances < Parent.rat) for distances in all_distances]}")
for index, distances in enumerate(all_distances):
    zone_condition = (distances > Parent.ral) & (distances < Parent.rat)
    print(zone_condition)
    selected_indices = target_indices[index][zone_condition]
    print(selected_indices)
    #selected_indices = np.where(zone_condition, target_indices[index], 0)
    #print(selected_indices)
    #selected_indices = remove_false_zeros(zone_condition, selected_indices)
   # print(selected_indices)
    
    #selected_indices = remove_self_index(index, selected_indices)
    #print(selected_indices)
    selected_distances = distances[zone_condition]#np.where(zone_condition, distances, 0)
    print(selected_distances)
    #selected_distances = selected_distances[selected_distances != 0]
   # print(selected_distances)

    selected_indices, mask = remove_hidden_indices(index, selected_indices, selected_distances)
    print(selected_indices)

    selected_distances = selected_distances[mask]
    print(selected_distances)
    cjs = pop.population_positions[selected_indices]
    print(cjs)
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
#print(pop.population_positions[:, np.newaxis, :])

'''zone_condition = (distances > Parent.ral) & (distances < Parent.rat)
print(zone_condition)

selected_indices = neighbours[zone_condition]#np.where(zone_condition, neighbours, 0)
print(selected_indices)

selected_distances = np.where(zone_condition, distances, 0)
print(selected_distances)

#selected_indices = np.apply_along_axis(remove_false_zeros, 1, zone_condition, selected_indices)
#selected_distances = np.apply_along_axis(lambda x: x[x != 0], 1, selected_distances)

def remove_hidden(index):
    return remove_hidden_indices(index, selected_indices[index], selected_distances[index])

hidden_results = np.array([remove_hidden(index) for index in range(pop.population_size)])
selected_indices = np.array([result[0] for result in hidden_results])
masks = np.array([result[1] for result in hidden_results])
selected_distances = np.array([selected_distances[index][masks[index]] for index in range(pop.population_size)])

cjs = np.array([pop.population_positions[indices] for indices in selected_indices])
pos = pop.population_positions[:, np.newaxis, :]
cj_minus_ci = cjs - pos
normalised = cj_minus_ci / selected_distances[:, :, np.newaxis]
sum_of_normalised = np.sum(normalised, axis=1)

all_repulsion_vectors = -sum_of_normalised
print(all_repulsion_vectors)'''