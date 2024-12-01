import numpy as np
import random
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
from prey_class import Prey
from environment_class import Environment
from population_class import Population

animation_on = False

P_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PolarisationData"
R_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\RotationData"

polarisation_data = np.load(f'{P_path}/polarisation_data.npy', allow_pickle=True)
rotation_data = np.load(f'{R_path}/rotation_data.npy', allow_pickle=True)
print(polarisation_data)

REPETITIONS = polarisation_data[0][1][3]
TIMESTEPS = polarisation_data[0][1][2]
time = np.linspace(0, TIMESTEPS, TIMESTEPS)
for r in range(REPETITIONS):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_ylim(-0.05,1.05)
    ax.plot(time, polarisation_data[r][0], label='Polarisation', c='red')
    ax.plot(time, rotation_data[r][0], label='Rotation', c='blue')
    ax.legend()
    ax.set_title(f'Repetition {r+1}')

plt.show()




'''
if animation_on:
    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    #ax1.set_axis_off()
    ax1.set_xlim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_ylim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_zlim3d(-env.radius*1.01, env.radius*1.01)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    scatter3D = ax1.scatter(np.array([pos[0] for pos in all_positions]),
                            np.array([pos[1] for pos in all_positions]), 
                            np.array([pos[2] for pos in all_positions]),
                            s=10,
                            c=['blue' if isinstance(agent, Prey) else 'red' for agent in pop.population_array])
    xyscatter = ax1.scatter(np.array([pos[0] for pos in all_positions]),
                            np.array([pos[1] for pos in all_positions]), 
                            np.full(pop.population_size,-env.radius), 
                            zdir='z', s=10, c='gray', alpha=0.4)
    xzscatter = ax1.scatter(np.array([pos[0]for pos in all_positions]), 
                            np.full(pop.population_size,env.radius), 
                            np.array([pos[2] for pos in all_positions]), 
                            zdir='y', s=10, c='gray', alpha=0.4)
    yzscatter = ax1.scatter(np.full(pop.population_size,-env.radius), 
                            np.array([pos[1] for pos in all_positions]), 
                            np.array([pos[2] for pos in all_positions]), 
                            zdir='x', s=10, c='gray', alpha=0.4)

for t in range(TIMESTEPS):            #Update the scatter plot for each timestep
    pop.update_positions(env)
    data_recorder.update_data(pop, time=t, repetitions=n)
    
    if graph_on:
        scatter3D._offsets3d = (np.array([pos[0] for pos in pop.population_positions]),
                                np.array([pos[1] for pos in pop.population_positions]), 
                                np.array([pos[2] for pos in pop.population_positions]))
        xyscatter._offsets3d = (np.array([pos[0] for pos in pop.population_positions]), 
                                np.array([pos[1] for pos in pop.population_positions]),
                                np.full(pop.population_size,-env.radius))
        xzscatter._offsets3d = (np.array([pos[0] for pos in pop.population_positions]),
                                np.full(pop.population_size, env.radius),
                                np.array([pos[2] for pos in pop.population_positions]))
        yzscatter._offsets3d = (np.full(pop.population_size,-env.radius),
                                np.array([pos[1] for pos in pop.population_positions]),
                                np.array([pos[2] for pos in pop.population_positions]))
        ax1.set_title(f'Time: {t}')
        
        plt.pause(0.001)
'''