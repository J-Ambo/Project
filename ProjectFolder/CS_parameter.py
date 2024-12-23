import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil

data_path = r"C:\Users\44771\Desktop\Data\2012\2012_1510"
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
X1 = x1 - 1
O_r1 = rotation_averages
O_p1 = polarisation_averages

'''strip_path1 = r"C:\\Users\\44771\\Desktop\\Data\\1712\\1712_1318"
Strip_Or_data1 = np.load(f'{strip_path1}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data1 = np.load(f'{strip_path1}/polarisation_averages.npy', allow_pickle=True)

strip_path2 = r"C:\\Users\\44771\\Desktop\\Data\\1712\\1712_1734"
Strip_Or_data2 = np.load(f'{strip_path2}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data2 = np.load(f'{strip_path2}/polarisation_averages.npy', allow_pickle=True)

strip_path3 = r"C:\\Users\\44771\\Desktop\\Data\\1912\\1912_1401"
Strip_Or_data3 = np.load(f'{strip_path3}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data3 = np.load(f'{strip_path3}/polarisation_averages.npy', allow_pickle=True)

fig, ax1 = plt.subplots(figsize=(5, 4))
ax1.set_title('$O_r$ vs $\Delta r_{al}$')
ax1.set_ylim(0, 1)
ax1.set_ylabel('$O_r$')
ax1.set_xlabel('$\Delta r_{al}$')
ax1.plot(X1, O_r1[0])
ax1.plot(X1, Strip_Or_data1.reshape(11))
ax1.plot(X1, Strip_Or_data2.reshape(11))
ax1.plot(X1, Strip_Or_data3.reshape(11))
ax1.invert_xaxis()

fig, ax2 = plt.subplots(figsize=(5, 4))
ax2.set_title('$O_p$ vs $\Delta r_{al}$')
ax2.set_ylim(0, 1)
ax2.set_ylabel('$O_p$')
ax2.set_xlabel('$\Delta r_{al}$')
ax2.plot(X1, O_p1[0])
ax2.plot(X1, Strip_Op_data1.reshape(11))
ax2.plot(X1, Strip_Op_data2.reshape(11))
ax2.plot(X1, Strip_Op_data3.reshape(11))
ax2.invert_xaxis()'''

strip_path2 = r"C:\\Users\\44771\\Desktop\\Data\\1712\\1712_1734"
Strip_Or_data2 = np.load(f'{strip_path2}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data2 = np.load(f'{strip_path2}/polarisation_averages.npy', allow_pickle=True)

fig, ax1 = plt.subplots(figsize=(5, 4))
ax1.set_title('$O_r$ vs $\Delta r_{al}$')
ax1.set_ylim(0, 1)
ax1.set_ylabel('$O_r$')
ax1.set_xlabel('$\Delta r_{al}$')
ax1.plot(X1, O_r1.reshape(11))
ax1.plot(X1, Strip_Or_data2.reshape(11))

ax1.invert_xaxis()

plt.show()
