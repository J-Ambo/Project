import numpy as np
from matplotlib import pyplot as plt
import os
import shutil
'''Plotting the polarisation and rotation data for each repetition.

   Also provides an alternative method to matplotlib.animation.FuncAnimation for creating
   an animation of the simulation. It uses plt.pause between each iteration to update the plot.
'''
animation_on = True

data_path = r"C:\Users\44771\Desktop\Data\2212\2212_1308"
data_file_name = os.path.split(data_path)[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
position_data = np.load(f'{data_path}/position_data.npy', allow_pickle=True)

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

# Pick which (s)trip, (i)ncrement, (r)epetition to animate
s = 4
i = 4
r = 7

# Animation
if animation_on:

    fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
    ax1.set_xlim3d(-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_ylim3d(-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_zlim3d(-Arena_radius*1.01, Arena_radius*1.01)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    scatter3D = ax1.scatter(np.array([pos[0] for pos in position_data[s][i][r][0]]),
                            np.array([pos[1] for pos in position_data[s][i][r][0]]), 
                            np.array([pos[2] for pos in position_data[s][i][r][0]]),
                            s=10,
                            c='blue')
    xyscatter = ax1.scatter(np.array([pos[0] for pos in position_data[s][i][r][0]]),
                            np.array([pos[1] for pos in position_data[s][i][r][0]]), 
                            np.full(int(Population_size),-Arena_radius), 
                            zdir='z', s=10, c='gray', alpha=0.4)
    xzscatter = ax1.scatter(np.array([pos[0]for pos in position_data[s][i][r][0]]), 
                            np.full(int(Population_size),Arena_radius), 
                            np.array([pos[2] for pos in position_data[s][i][r][0]]), 
                            zdir='y', s=10, c='gray', alpha=0.4)
    yzscatter = ax1.scatter(np.full(int(Population_size),-Arena_radius), 
                            np.array([pos[1] for pos in position_data[s][i][r][0]]), 
                            np.array([pos[2] for pos in position_data[s][i][r][0]]), 
                        zdir='x', s=10, c='gray', alpha=0.4)

    for t in range(int(Timesteps)):
        scatter3D._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]),
                                np.array([pos[1] for pos in position_data[s][i][r][t]]), 
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        xyscatter._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]), 
                                np.array([pos[1] for pos in position_data[s][i][r][t]]),
                                np.full(int(Population_size),-Arena_radius))
        xzscatter._offsets3d = (np.array([pos[0] for pos in position_data[s][i][r][t]]),
                                np.full(int(Population_size), Arena_radius),
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        yzscatter._offsets3d = (np.full(int(Population_size),-Arena_radius),
                                np.array([pos[1] for pos in position_data[s][i][r][t]]),
                                np.array([pos[2] for pos in position_data[s][i][r][t]]))
        ax1.set_title(f'Time: {t} Strip: {s+1} Increment: {i+1} Repetition: {r+1}')

        #plt.savefig(f'C:/Users/44771/Desktop/GifImages/Rep{n+1}Time{t}.png')
        
        plt.pause(0.001)
