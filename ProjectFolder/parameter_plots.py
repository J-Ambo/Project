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

data_path = r"C:\Users\44771\Desktop\Data\1712\1712_0846"
data_file_name = os.path.split(data_path)[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]]

## Other data ##
#data_path2 = r"C:\Users\44771\Desktop\Data\1512\1512_0308"
#polarisation_data2 = np.load(f'{data_path2}/polarisation_data.npy', allow_pickle=True)
#polarisation_averages2 = np.load(f'{data_path2}/polarisation_averages.npy', allow_pickle=True)
#rotation_averages = np.load(f'{data_path2}/rotation_averages.npy', allow_pickle=True)
#ral_array2 = [sub_array[-2] for sub_array in polarisation_data2[0][:,0][:,1]]
#rat_array2 = [sub_array[-1] for sub_array in polarisation_data2[0][:,0][:,1]]

x1 = np.asarray(ral_array) #np.asarray(ral_array2)
y1 = np.asarray(rat_array) #np.asarray(rat_array2)

X1 = x1 - 1 #x2 - 1
Y1 = y1 - x1[0] #y2 - x2
O_r1 = rotation_averages
O_p1 = polarisation_averages

O_i = [O_p1, O_r1]
label = ['$O_p$', '$O_r$']
colour = ['b', 'r']

for i in range(2):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))

    ax.plot_wireframe(X1, np.tile(Y1, (11,1)).T, O_i[i], rstride=1, cstride=1, color=colour[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(label[i])
    ax.set_zlim(0, 1)
    ax.invert_xaxis()

    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{time_dm}/{data_file_name}'
    if save_plots:
        plt.savefig(f'{new_folder_path}/{label[i]}', dpi=300, bbox_inches=None)

strip_path1 = r"C:\Users\44771\Desktop\Data\1712\1712_1318"
Strip_Or_data1 = np.load(f'{strip_path1}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data1 = np.load(f'{strip_path1}/polarisation_averages.npy', allow_pickle=True)

strip_path2 = r"C:\Users\44771\Desktop\Data\1712\1712_1734"
Strip_Or_data2 = np.load(f'{strip_path2}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data2 = np.load(f'{strip_path2}/polarisation_averages.npy', allow_pickle=True)

fig, ax1 = plt.subplots(figsize=(5, 4))
ax1.plot(X1, O_r1[0])
ax1.plot(X1, Strip_Or_data1.reshape(11))
ax1.plot(X1, Strip_Or_data2.reshape(11))
ax1.invert_xaxis()

fig, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(X1, O_p1[0])
ax2.plot(X1, Strip_Op_data1.reshape(11))
ax2.plot(X1, Strip_Op_data2.reshape(11))
ax2.invert_xaxis()

plt.show()




