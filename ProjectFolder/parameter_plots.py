import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil
'''Plotting the polarisation and rotation order parameters as functions of
    the alignment and attraction zone widths.
'''
plots_on = True
save_plots = False


data_path = r"C:\Users\44771\Desktop\Data\2212\2212_1308"
data_file_name1 = os.path.split(data_path)[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]]

x1 = np.asarray(ral_array)
y1 = np.asarray(rat_array) 
X1 = x1 - 1 
Y1 = y1 - x1[0]
O_r1 = rotation_averages
O_p1 = polarisation_averages

O_i1 = [O_p1, O_r1]
label = ['$O_p$', '$O_r$']
colour = ['b', 'r']

for i in range(2):

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))
    ax.plot_wireframe(X1, np.tile(Y1, (11,1)).T, O_i1[i], rstride=1, cstride=1, color=colour[i])
    ax.set_xlabel('$\Delta r_{al}$')
    ax.set_ylabel('$\Delta r_{at}$')
    ax.set_zlabel(label[i])
    ax.set_zlim(0, 1)
    ax.invert_xaxis()
    #plt.savefig(f"C:/Users/44771/Desktop/GifImages/pan/{label[i]}_rot_{elev}_{azim}.png")

    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{time_dm}/{data_file_name1}'
    if save_plots:

        plot_dir = f'{new_folder_path}'#
        os.makedirs(plot_dir, exist_ok=True)#
        #plt.savefig(f'{plot_dir}/{label[i]}_{data_file_name2}-{data_file_name3}-{data_file_name1}', dpi=300, bbox_inches=None)#
        plt.savefig(f'{new_folder_path}/{label[i]}', dpi=300, bbox_inches=None)

plt.show()