import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil
'''Plotting the polarisation and rotation order parameters as functions of
    the alignment and attraction zone widths.
'''
plots_on = False
save_plots = False

data_path = r"C:\Users\44771\Desktop\Data\1512\1512_1647"
data_file_name = os.path.split(data_path)[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)
#print(polarisation_averages)
#print(rotation_averages)
Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array1 = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array1 = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]]

## Other data ##
data_path2 = r"C:\Users\44771\Desktop\Data\1512\1512_0308"
polarisation_data2 = np.load(f'{data_path2}/polarisation_data.npy', allow_pickle=True)
polarisation_averages2 = np.load(f'{data_path2}/polarisation_averages.npy', allow_pickle=True)
#rotation_averages = np.load(f'{data_path2}/rotation_averages.npy', allow_pickle=True)
ral_array2 = [sub_array[-2] for sub_array in polarisation_data2[0][:,0][:,1]]
rat_array2 = [sub_array[-1] for sub_array in polarisation_data2[0][:,0][:,1]]


x1, x2 = np.asarray(ral_array1), np.asarray(ral_array2)
y1, y2 = np.asarray(rat_array1), np.asarray(rat_array2)

X1, X2 = x1 - 1, x2 - 1
Y1, Y2 = y1 - x1, y2 - x2
O_r1 = rotation_averages
O_p1 = polarisation_averages

O_p2 = polarisation_averages2
print(x2)
print(np.tile(x2, (16,1)).T)
print(np.tile(y2, (16,1)).T.shape)
print(O_p2.shape)
print(Y2)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(5, 4))
ax.plot_wireframe(X1, Y1, O_r1, rstride=10, cstride=10, color='r')
ax.plot_wireframe(X1, Y1, O_p1, rstride=10, cstride=10, color='b')

#X2, Y2 = np.meshgrid(X2, Y2)
ax.plot_wireframe(x2, np.tile(y2, (16,1)).T, O_p2, rstride=1, cstride=1, color='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_zlim(0, 1)

ax.invert_xaxis()
plt.show()

