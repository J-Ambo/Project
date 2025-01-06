import numpy as np
from matplotlib import pyplot as plt
from matplotlib import text as txt
import time
import os
import shutil

save_plots = False

data_path = r"C:\Users\44771\Desktop\Data\0101\0101_0700"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
print(data_file_name2)

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)
p_errors = np.load(f'{data_path}/polarisation_errors.npy', allow_pickle=True)
r_errors = np.load(f'{data_path}/rotation_errors.npy', allow_pickle=True)

print(polarisation_averages)
print(len(polarisation_averages))

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]]

x1 = np.asarray(ral_array)
X1 = x1 - 1
print(X1)
#print(np.tile(X1, (11,1)))
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

##
'''strip_path2 = r"C:\\Users\\44771\\Desktop\\Data\\1712\\1712_1734"
Strip_Or_data2 = np.load(f'{strip_path2}/rotation_averages.npy', allow_pickle=True)
Strip_Op_data2 = np.load(f'{strip_path2}/polarisation_averages.npy', allow_pickle=True)'''

label = ['$O_r$', '$O_p$']
data = [O_r1, O_p1]
errs = [r_errors, p_errors]
lb = [20, 30, 40, 50, 60]
#lb = [360, 330, 300, 270, 240]
#lb = [0.05, 0.10, 0.15, 0.20, 0.25]
#lb =[240]
locn = ['upper right', 'upper right']
colours = [plt.get_cmap('Reds')(np.linspace(0.3, 1, 5)), plt.get_cmap('Blues')(np.linspace(0.3, 1, 5))]

time_dm = time.strftime('%d%m')
new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'#{time_dm}/{data_file_name1}'
os.makedirs(new_folder_path, exist_ok=True)
shutil.copy(f'{data_path}/parameters.txt', new_folder_path)

plt.rcParams.update({"text.usetex": True, "font.family": "Nimbus Roman"})
for i in range(2):
    fig, ax1 = plt.subplots(figsize=(6,5))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(label[i], size=16.0)
    ax1.set_xlabel('$\Delta r_{al}$', size=16.0)
    ax1.tick_params(labelsize=14)
    for j in range(len(polarisation_averages)):
        ax1.errorbar(X1, data[i][j], yerr=errs[i][j], capsize=2, color=colours[i][j], label=f'{lb[j]}$^\circ s^{{-1}}$') #\sigma$')# #s^{{-1}}$')
    ax1.invert_xaxis()
    ax1.legend(frameon=False, fontsize=14, loc=locn[i], labelspacing=0.3, handlelength=1.8, handletextpad=0.5)

    if save_plots:
        plt.savefig(f'{new_folder_path}/{label[i]}_CS', dpi=300, bbox_inches=None)


#ax1.plot(X1, O_r1.reshape(11))
#ax1.plot(X1, Strip_Or_data2.reshape(11))

plt.show()
