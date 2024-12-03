########################################
##     Tests                          ##
########################################

from environment_class import Environment
from population_class import Population
import numpy as np

env = Environment(10, 3)
pop = Population(10, 5, env)
pop.population_array 
indices_array = np.array([0, 1 ,2, 3, 4, 5, 6, 7, 8, 9])
select = np.array([1,2,5,7])

select_indices = np.isin(indices_array, select) #returns a boolean array indicating which elements are in the select array
where = np.where(select_indices[:,np.newaxis], pop.population_positions, 0)
pos = pop.population_positions[0]
print(select_indices)
print(where)
cj_minus_c0 = np.where(select_indices[:,np.newaxis], where - pos, 0)
print(cj_minus_c0)
normals = np.linalg.norm(cj_minus_c0, axis=1)
print(normals[:,np.newaxis])
normalised = np.where(select_indices[:,np.newaxis], cj_minus_c0/(normals[:,np.newaxis] + 1e-6), cj_minus_c0)
print(normalised)
normalised_sum = np.sum(normalised, axis=0)
print(normalised_sum)

#print(np.tile(indices_array, (len(indices_array),1)))
