########################################
##     Tests                          ##
########################################

from environment_class import Environment
from population_class import Population
import numpy as np

test_prints = True
env = Environment(10, 3)
pop = Population(10, 5, env)
indices_array = np.arange(10)
select = np.array([1,2,5,7])



target_indices = np.array([np.array([2,3]), np.array([1,3,6,9]), np.array([1,2,3]), np.array([1,2,3,4,5,9]), np.array([7]), 
                           np.array([3,5,6]), np.array([2,3]), np.array([1,9]), np.array([8]), np.array([6,7])], dtype=object)

for index, target in enumerate(target_indices):
    select_indices = np.isin(indices_array, target) # returns a boolean array indicating which elements are in the target array
    where = np.where(select_indices[:,np.newaxis], pop.population_positions, 0)
    pos = pop.population_positions[index]
    select_positions = np.where(select_indices[:,np.newaxis], pop.population_positions, 0)
    cj_minus_c0 = np.where(select_indices[:,np.newaxis], where - pos, 0)
    normals = np.linalg.norm(cj_minus_c0, axis=1)
    normalised = np.where(select_indices[:,np.newaxis], cj_minus_c0/(normals[:,np.newaxis] + 1e-6), cj_minus_c0)
    normalised_sum = np.sum(normalised, axis=0)

    repulsion_vector = -normalised_sum
    
    if test_prints:
        print(f"Target: \n{target}")
        print(f"Boolean: \n{select_indices}")
        print(f"Selected positions: \n{where}")
        print(f"Position, ci: \n{pos}")
        print(f"cj_minus_c0: \n{cj_minus_c0}")
        print(f"Normals: \n{normals[:,np.newaxis]}")
        print(f"Normalised: \n{normalised}")
        print(f"Sum: \n{normalised_sum}")

#print(np.tile(indices_array, (len(indices_array),1)))

truth_array = np.full(len(indices_array), True)
target_indices = np.array([np.array([2,3]), np.array([1,3,6,9]), np.array([1,2,3])], dtype=object)

#print(np.where(truth_array[:,np.newaxis], target_indices, 0))
tiles = np.tile(indices_array, (len(target_indices),1))
targets_positions = np.isin(indices_array, target_indices.reshape(len(target_indices),1)[0])
#print(np.where(tiles, ))

#print(target_indices.reshape(len(target_indices),1)[0])
#print(f"Targets: {targets_positions}")
print(np.tile(np.array([np.arange(10), np.arange(4)], dtype=object), (10,1)))
