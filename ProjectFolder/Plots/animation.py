import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
from sklearn.neighbors import LocalOutlierFactor

'''Plotting the polarisation and rotation data for each repetition.

   Also provides an alternative method to matplotlib.animation.FuncAnimation for creating
   an animation of the simulation. It uses plt.pause between each iteration to update the plot.
'''
animation_on = True

data_path = r"C:\Users\44771\Desktop\Data\2403\2403_1117\Sp3.4_1366653375"
data_file_name = os.path.split(data_path)[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
position_data = np.load(f'{data_path}/position_data.npy', allow_pickle=True)
predator_positions = np.load(f'{data_path}/predator_positions.npy', allow_pickle=True)

parameters = {}
with open(f'{data_path}/parameters.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            parameters[key.strip()] = value.strip()

strips = int(parameters['Strips'])
increments = int(parameters['Increments'])
repetitions = int(parameters['Repetitions'])
timesteps = int(parameters['Timesteps'])
population = int(parameters['Population'])
arena = int(parameters['Arena Radius'])

# Pick which (s)trip, (i)ncrement, (r)epetition to animate
s = 0
i = 0
r = 0


# Animation
if animation_on:

    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax1.view_init(45, 45)
    ax1.set_xlim3d(-90, 90)    #-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_ylim3d(-90,90)    #-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_zlim3d(-90,90)    #-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    x_positions = np.array([pos[0] for pos in position_data[s][i][r][0]])
   # print(x_positions)
   # print(np.array([pos[0] for pos in position_data[s][i][r][1]]))
    y_positions = np.array([pos[1] for pos in position_data[s][i][r][0]])
    z_positions = np.array([pos[2] for pos in position_data[s][i][r][0]])
    scatter3D = ax1.scatter(x_positions,
                            y_positions, 
                            z_positions,
                            s=10,
                            c='blue')
    predator_scatter = ax1.scatter(predator_positions[s][i][r][0][0][0],
                                   predator_positions[s][i][r][0][0][1],
                                   predator_positions[s][i][r][0][0][2],
                                   s=10,
                                   c='red')
    xyscatter = ax1.scatter(np.array([pos[0] for pos in position_data[s][i][r][0]]),
                            np.array([pos[1] for pos in position_data[s][i][r][0]]), 
                            np.full(population,-arena), 
                            zdir='z', s=10, c='gray', alpha=0.4)
    xzscatter = ax1.scatter(np.array([pos[0]for pos in position_data[s][i][r][0]]), 
                            np.full(population,arena), 
                            np.array([pos[2] for pos in position_data[s][i][r][0]]), 
                            zdir='y', s=10, c='gray', alpha=0.4)
    yzscatter = ax1.scatter(np.full(population,-arena), 
                            np.array([pos[1] for pos in position_data[s][i][r][0]]), 
                            np.array([pos[2] for pos in position_data[s][i][r][0]]), 
                            zdir='x', s=10, c='gray', alpha=0.4)

    for t in range(timesteps):
        scatter3D._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]),
                                np.array([pos[1] for pos in position_data[s][i][r][t]]), 
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        predator_scatter._offsets3d = (np.array([predator_positions[s][i][r][t][0][0]]),
                                        np.array([predator_positions[s][i][r][t][0][1]]), 
                                        np.array([predator_positions[s][i][r][t][0][2]]))
        xyscatter._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]), 
                                np.array([pos[1] for pos in position_data[s][i][r][t]]),
                                np.full(population,-arena))
        xzscatter._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]),
                                np.full(population, arena),
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        yzscatter._offsets3d = (np.full(population,-arena),
                                np.array([pos[1] for pos in position_data[s][i][r][t]]),
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        ax1.set_title(f'Time: {t} Strip: {s+1} Increment: {i+1} Repetition: {r+1}')

        #plt.savefig(f'C:/Users/44771/Desktop/GifImages/Rep{n+1}Time{t}.png')
        
        plt.pause(0.2)

