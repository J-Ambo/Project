########################################
##     Tests                          ##
########################################

from AgentClasses.environment_class import Environment
from AgentClasses.population_class import Population
from AgentClasses.parent_class import Parent
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib.animation import FuncAnimation
from AgentClasses.predator_class import Predator

Parent.ral = 2
Parent.rat = 15
Population.steering_error = 0.25
Parent.perception_angle = np.deg2rad(270)
Parent.maximal_turning_angle = np.deg2rad(40)
Parent.evasion_angle = np.deg2rad(20)
Parent.speed = 2
Population.selfish = 0

a= np.array([[1,2]])
b= np.array([[3,2]])
c = np.array([[0,0], [1,1], [100,4]])
c[:-1] += np.array([2,2])
print(c)
#print(np.concatenate([a, b], axis=0))
test_prints = False
env = Environment(10, 3,5)
predator = Predator(0,0,-200)
pop = Population(50, env, predator)
indices_array = np.arange(10)
select = np.array([1,2,5,7])



target_indices = np.array([np.array([]), np.array([1,3,6,9]), np.array([1,2,3]), np.array([1,2,3,4,5,9]), np.array([7]), 
                           np.array([3,5,6]), np.array([2,3]), np.array([1,9]), np.array([8]), np.array([6,7])], dtype=object)

all_repulsion_vectors = np.empty((pop.population_size, 3))
'''for index, target in enumerate(target_indices):
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
        print(f"Sum: \n{normalised_sum}")'''
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
Neighbours = tree.query(all_positions, k=9)
distances = np.asarray(Neighbours[0])
neighbours = np.asarray(Neighbours[1])
#print(f"distances: \n{distances}")
#print(f"neighbours: \n{neighbours}")
rat_neighbours = tree.query_radius(all_positions, Parent.rat, return_distance=True)
print("rat neighbours \n", rat_neighbours)

length_array = np.zeros(pop.population_size)
distance_array = np.zeros(pop.population_size)
for n in range(pop.population_size):
     nN = len(rat_neighbours[0][n])-1
     d = rat_neighbours[1][n][-1]
    # print(nN, d)
     density = nN/d

#print(len(rat_neighbours[0])-1)
ral_neighbours = tree.query_radius(all_positions, Parent.ral)
rr_neighbours = tree.query_radius(all_positions, Parent.rr)

densities = 1 / distances[:, -1]
#print(densities)
#distances = rat_neighbours[1]
self_mask = distances != 0
#print(self_mask)
#print("masked rat neighbours \n", rat_neighbours[0][self_mask])

# Find the set difference between these arrays
rz_Nei = rr_neighbours
#alz_Nei = [np.setdiff1d(ral, rr) for ral, rr in zip(ral_neighbours, rr_neighbours)]
#atz_Nei = [np.setdiff1d(rat, ral) for rat, ral in zip(rat_neighbours, ral_neighbours)]



target_indices = rat_neighbours[0]
all_distances = rat_neighbours[1]

'''def remove_false_zeros(boolean, indices):    valid_indices = []
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


import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Generate sample 3D data
np.random.seed(0)
n_points = 100
n_outliers = 10
points = np.random.normal(0, 1, size=(n_points, 3))
outliers = np.random.normal(5, 2, size=(n_outliers, 3))
points = np.concatenate((points, outliers))

# K-Nearest Neighbors (KNN) density estimation
knn = NearestNeighbors(n_neighbors=10)
knn.fit(points)
distances, _ = knn.kneighbors(points)
densities = 1 / distances[:, -1]
#print("distances \n",distances)
#print("distances[:,-1] \n", distances[:,-1])
#print(densities)

'''# Equivalence classes
n_classes = 5
density_bins = np.linspace(densities.min(), densities.max(), n_classes + 1)
class_labels = np.digitize(densities, density_bins)
print(density_bins)
print(class_labels)

# Outlier detection
outlier_threshold = 0.09
outlier_labels = class_labels <= np.percentile(class_labels, outlier_threshold * 100)
outliers = points[outlier_labels]
print(outliers)

# Visualize the results
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[~outlier_labels, 0], points[~outlier_labels, 1], points[~outlier_labels, 2], c='b', alpha=0.5)
ax.scatter(points[outlier_labels, 0], points[outlier_labels, 1], points[outlier_labels, 2], c='r', alpha=0.5)
plt.show()'''


import os
data_path = r"C:\Users\44771\Desktop\Data\0101\0101_0700"
data_file_name = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
##print(data_file_name2)

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
rotation_data = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)

samples=2
increments=1
strips=1
repetitions = 10
rotation_errors = np.zeros((strips, increments))
polarisation_errors = np.zeros((strips, increments))

##print(rotation_data[0][0])
##print(rotation_data[0][0][:,0])
rotation_sample = [repetition[-samples:] for repetition in rotation_data[0][0][:,0]]
#print(rotation_sample)
##print(np.std(rotation_sample) / np.sqrt(samples))

def pooled_standard_error(data, samples):
    n = samples
    dof = samples - 1
    incs = [repetition[-samples:] for repetition in rotation_data[0][0][:,0]]
    vars = np.var(incs, ddof=1, axis=1)
    pooled_stdev = np.sqrt(np.sum(vars)/repetitions)
    pooled_se = pooled_stdev*np.sqrt(repetitions/samples)
    return pooled_se

# Example usage:
pooled_se = pooled_standard_error(rotation_data, samples)
##print(f"Pooled Standard Error: {pooled_se}")

for s in range(strips):
    for i in range(increments):
        rotation_samples = [repetition[-samples:] for repetition in rotation_data[s][i][:,0]]
        polarisation_samples = [repetition[-samples:] for repetition in polarisation_data[s][i][:,0]]
        Rvars = np.var(rotation_samples, ddof=1, axis=1)
        Pvars = np.var(polarisation_samples, ddof=1, axis=1)

        Rpooled_stdev = np.sqrt(np.sum(Rvars)/repetitions)
        Ppooled_stdev = np.sqrt(np.sum(Pvars)/repetitions)

        Rpooled_se = Rpooled_stdev*np.sqrt(repetitions/samples)
        Ppooled_se = Ppooled_stdev*np.sqrt(repetitions/samples)
        
        rotation_errors[s][i] = Rpooled_se
        polarisation_errors[s][i] = Ppooled_se
        #rotation_samples = [repetition[-samples:] for repetition in rotation_data[s][i][:,0]]
        #rotation_errors[s][i] = np.std(rotation_samples) / np.sqrt(samples)

        #polarisation_samples = [repetition[-samples:] for repetition in polarisation_data[s][i][:,0]]
        #polarisation_errors[s][i] = np.std(polarisation_samples) / np.sqrt(samples)
##print(rotation_errors)
#new_folder_path = f'C:/Users/44771/Desktop/Data/{data_file_name2}/{data_file_name}'
#os.makedirs(new_folder_path, exist_ok=True)

#np.save(f'{data_path}/polarisation_errors', polarisation_errors)
#np.save(f'{data_path}/rotation_errors', rotation_errors)

s = 0
r = 0
i = 0

fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))

ax1.set_xlim3d(-50, 50)      #ax1.set_xlim3d(0, 38)
ax1.set_ylim3d(-50, 50)     #ax1.set_ylim3d(-10*0.9, 20*0.9)
ax1.set_zlim3d(-50, 50)           #ax1.set_zlim3d(0, 20*0.9)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')

#ax1.xaxis.set_major_formatter('')
#ax1.yaxis.set_major_formatter('')
#ax1.zaxis.set_major_formatter('')
ax1.set_proj_type('persp', focal_length=0.4)

#d_path = r"C:\Users\44771\Desktop\Data\2002\2002_1712"
d_path = r"C:\Users\44771\Desktop\Data\2803\2803_2251\Sp4.0_2285702475"        #Sp4.5_1489197482"
data_file_name1 = os.path.split(d_path)[1]
data_file_name2 = os.path.split(os.path.split(d_path)[0])[1]

position_data = np.load(f'{d_path}/position_data.npy', allow_pickle=True)
direction_data = np.load(f'{d_path}/direction_data.npy', allow_pickle=True)
##print(position_data[s][i][r])
x_positions = position_data[s][i][r][:,:,0]
y_positions = position_data[s][i][r][:,:,1]
z_positions = position_data[s][i][r][:,:,2]
##print(x_positions[0], y_positions[0], z_positions[0], position_data[0][0][1][0])
x_directions = direction_data[s][i][r][:,:,0]
y_directions = direction_data[s][i][r][:,:,1]
z_directions = direction_data[s][i][r][:,:,2]

def get_densities(tree):
        distances = tree.query(position_data[s][i][r][t], k=10)[0]
        densities = np.zeros(50)
        for n in range(50):
            nN = len(distances[n])-1
            d = distances[n][-1]
            density = nN/d
            densities[n] = density
        return densities
def get_outlier_labels(tree):
        densities = get_densities(tree)
        n_classes = 100
        density_bins = np.linspace(densities.min(), densities.max(), n_classes + 1)
        class_labels = np.digitize(densities, density_bins)
        # Outlier detection
        outlier_threshold = 0.3
        outlier_labels = class_labels <= np.percentile(class_labels, outlier_threshold * 100)
        return outlier_labels

quiver = ax1.quiver(x_positions[0], y_positions[0], z_positions[0],
                       x_directions[0], y_directions[0], z_directions[0],
                       length=5,
                       arrow_length_ratio=0.2,
                       linewidth=1,
                       pivot='middle')
#xyscatter = ax1.scatter(x_positions[0],
 #                       y_positions[0], 
  #                      np.full(300,0), 
   #                     zdir='z', s=10, c='gray', alpha=0.4)
xzscatter = ax1.scatter(x_positions[0], 
                        np.full(300, 113),  #-12), 
                        z_positions[0], 
                        zdir='y', s=10, c='gray', alpha=0.4)
yzscatter = ax1.scatter(np.full(300,50),   #0), 
                        y_positions[0], 
                        z_positions[0], 
                        zdir='x', s=10, c='gray', alpha=0.4)
elev = 20
azim = np.linspace(30,210, 200)
ax1.view_init(elev, 52)

save_animation = False
if save_animation:
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}/FramesP'
    os.makedirs(new_folder_path, exist_ok=True)

for t in range(500):
   # tree = KDTree(position_data[s][i][r][t])
   # outlier_labels = get_outlier_labels(tree)
    #print(outlier_labels)
    ##outlier_positions = position_data[s][i][r][t][outlier_labels]
    #print(outlier_positions)
    ax1.set_title(f'Timestep = {t+1}', size=22, name='Times New Roman')
    quiver.remove()
    #xyscatter.remove()
    yzscatter.remove()
    xzscatter.remove()
    quiver = ax1.quiver(x_positions[t], y_positions[t], z_positions[t],
                        x_directions[t], y_directions[t], z_directions[t],
                        length=2,
                        arrow_length_ratio=0.3,
                        pivot='middle',
                        colors=['b' ],  #if outlier else 'b' for outlier in outlier_labels],
                        linewidths=1)
    #xyscatter = ax1.scatter(x_positions[t], 
     #                       y_positions[t], 
      #                      np.full(300,0), 
       #                     zdir='z', s=10, c='gray', alpha=0.4)
   # xzscatter.set_offsets(np.c_[x_positions[t], np.full(300, -12)])
    xzscatter = ax1.scatter(x_positions[t], 
                            np.full(300, 113),  #-12), 
                            z_positions[t], 
                             s=10, c='gray', alpha=0.4)
    yzscatter = ax1.scatter(np.full(300,50),   #0), 
                            y_positions[t], 
                            z_positions[t], 
                             s=10, c='gray', alpha=0.4)
    #ax1.set_title(f'Time: {t} Strip: {s+1} Increment: {i+1} Repetition: {r+1}')# Outliers: {len(outlier_positions)}')
   ## ax1.view_init(elev, azim[t])

    if save_animation:
        plt.savefig(f'{new_folder_path}/frame_{t}.png', dpi=300)
  
    plt.pause(0.05)

#print(x_positions[199], y_positions[199], z_positions[199])

angles = np.linspace(0, 90, 10)
#print(angles)
angles = np.deg2rad(angles)
u_x = np.cos(angles)
u_y = np.sin(angles)
u_xy = np.stack((u_x, u_y))
#print(u_x, u_y)
#print(u_xy.T)
#print(u_xy.T[:,0]**2 + u_xy.T[:,1]**2)

a = np.array([[1,2], [2,3], [1,1]])
b = np.array([1,2])
#print(b.shape, a.shape)
#print(np.dot(a, b))


from numpy.random import PCG64, Generator
import time

s = time.perf_counter()
bit_gen = PCG64()#seed=12345)
rng = Generator(bit_gen)

integers = rng.integers(0, 2**32 -1, size=10)
f = time.perf_counter()
print(f'Finished in {f-s}')
print(integers)

speeds = np.linspace(0,5, 10)
zipped = list(zip(speeds, integers))
args, seed = zip(*zipped)
#print(zipped)

#print(os.listdir(r'C:\Users\44771\Documents\Level4Project\ProjectFolder\Plots'))
#for i in os.listdir(r'C:\Users\44771\Documents\Level4Project\ProjectFolder\Plots'):
#     print(i)
        
print([i for i in range(10)])