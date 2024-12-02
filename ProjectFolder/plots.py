import numpy as np
from matplotlib import pyplot as plt
import time
'''Plotting the polarisation and rotation data for each repetition.

   Also provides an alternative method to matplotlib.animation.FuncAnimation for creating
   an animation of the simulation. It uses plt.pause between each iteration to update the plot.
'''

animation_on = True

Pol_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PolarisationData"
R_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\RotationData"
Pos_path = r"C:\Users\44771\Documents\Level4Project\ProjectFolder\PositionData"

polarisation_data = np.load(f'{Pol_path}/polarisation_data.npy', allow_pickle=True)
rotation_data = np.load(f'{R_path}/rotation_data.npy', allow_pickle=True)
position_data = np.load(f'{Pos_path}/position_data.npy', allow_pickle=True)
POPULATION, NEIGHBOURS, ARENA_RADIUS, TIMESTEPS, DIMENSIONS, REPETITIONS, ral = (polarisation_data[0][1][0], polarisation_data[0][1][1],
                                                                             polarisation_data[0][1][2], polarisation_data[0][1][3], 
                                                                             polarisation_data[0][1][4], polarisation_data[0][1][5],
                                                                             polarisation_data[0][1][6])

print(int(TIMESTEPS))
time_steps = np.linspace(0, int(TIMESTEPS), int(TIMESTEPS))
for r in range(int(REPETITIONS)):
    fig, ax = plt.subplots(figsize=(4,3))
    ax.set_ylim(-0.05,1.05)
    ax.plot(time_steps, polarisation_data[r][0], label='Polarisation', c='red')
    ax.plot(time_steps, rotation_data[r][0], label='Rotation', c='blue')
    ax.legend()
    ax.set_title(f'Rep:{r+1} Pop:{int(POPULATION)} Nn:{int(NEIGHBOURS)} ral:{ral}')

plt.show()

if animation_on:
    for n in range(int(REPETITIONS)):
        fig, ax1 = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(7, 7))
        #ax1.set_axis_off()
        ax1.set_xlim3d(-ARENA_RADIUS*1.01, ARENA_RADIUS*1.01)
        ax1.set_ylim3d(-ARENA_RADIUS*1.01, ARENA_RADIUS*1.01)
        ax1.set_zlim3d(-ARENA_RADIUS*1.01, ARENA_RADIUS*1.01)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        scatter3D = ax1.scatter(np.array([pos[0] for pos in position_data[n][0]]),
                                np.array([pos[1] for pos in position_data[n][0]]), 
                                np.array([pos[2] for pos in position_data[n][0]]),
                                s=10,
                                c='blue')
        xyscatter = ax1.scatter(np.array([pos[0] for pos in position_data[n][0]]),
                                np.array([pos[1] for pos in position_data[n][0]]), 
                                np.full(int(POPULATION),-ARENA_RADIUS), 
                                zdir='z', s=10, c='gray', alpha=0.4)
        xzscatter = ax1.scatter(np.array([pos[0]for pos in position_data[n][0]]), 
                                np.full(int(POPULATION),ARENA_RADIUS), 
                                np.array([pos[2] for pos in position_data[n][0]]), 
                                zdir='y', s=10, c='gray', alpha=0.4)
        yzscatter = ax1.scatter(np.full(int(POPULATION),-ARENA_RADIUS), 
                                np.array([pos[1] for pos in position_data[n][0]]), 
                                np.array([pos[2] for pos in position_data[n][0]]), 
                            zdir='x', s=10, c='gray', alpha=0.4)
    
        for t in range(TIMESTEPS):
            scatter3D._offsets3d = (np.array([pos[0] for pos in position_data[n][t]]),
                                    np.array([pos[1] for pos in position_data[n][t]]), 
                                    np.array([pos[2] for pos in position_data[n][t]]))
            xyscatter._offsets3d = (np.array([pos[0] for pos in position_data[n][t]]), 
                                    np.array([pos[1] for pos in position_data[n][t]]),
                                    np.full(int(POPULATION),-ARENA_RADIUS))
            xzscatter._offsets3d = (np.array([pos[0] for pos in position_data[n][t]]),
                                    np.full(int(POPULATION), ARENA_RADIUS),
                                    np.array([pos[2] for pos in position_data[n][t]]))
            yzscatter._offsets3d = (np.full(int(POPULATION),-ARENA_RADIUS),
                                    np.array([pos[1] for pos in position_data[n][t]]),
                                    np.array([pos[2] for pos in position_data[n][t]]))
            ax1.set_title(f'Time: {t}')

            #plt.savefig(f'C:/Users/44771/Desktop/GifImages/Rep{n+1}Time{t}.png')
            
            plt.pause(0.001)
        plt.close()
        time.sleep(5)
    
