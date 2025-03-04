import numpy as np
from matplotlib import pyplot as plt
from matplotlib import text as txt
import time
import os
import shutil

save_plots = False

data_path = r"C:\Users\44771\Desktop\Data\0101\0101_0132"
#data_path = r"C:\Users\44771\Desktop\Data\0101\0101_0700"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
#print(data_file_name2)

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_samples = [repetition[-1000:] for repetition in polarisation_data[0][0][:,0]]
average_polarisation = np.mean(polarisation_samples)
print( average_polarisation)

polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)
p_errors = np.load(f'{data_path}/polarisation_errors.npy', allow_pickle=True)
r_errors = np.load(f'{data_path}/rotation_errors.npy', allow_pickle=True)

print(polarisation_averages)
#print(len(polarisation_averages))

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

label = ['$O_\mathrm{r}$', '$O_\mathrm{p}$']
data = [O_r1, O_p1]
errs = [r_errors, p_errors]
#lb = [3.0, 3.5, 4.0, 4.5, 5.0]
#lb = [20, 30, 40, 50, 60]
lb = [360, 330, 300, 270, 240]
#lb = [0.05, 0.10, 0.15, 0.20, 0.25]
#lb =[240]
locn = ['upper right', 'upper right']
colours = [plt.get_cmap('Reds')(np.linspace(0.3, 1, 5)), plt.get_cmap('Blues')(np.linspace(0.3, 1, 5))]

time_dm = time.strftime('%d%m')
new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'#{time_dm}/{data_file_name1}'
os.makedirs(new_folder_path, exist_ok=True)
shutil.copy(f'{data_path}/parameters.txt', new_folder_path)

plt.rcParams.update({"text.usetex": True, "font.family": "Nimbus Roman"})
fn = ['O_r', 'O_p']
for i in range(2):
    fig, ax1 = plt.subplots(figsize=(6,5))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(label[i], size=22.0)
    ax1.set_xlabel('$\Delta r_\mathrm{al}$', size=22.0)
    ax1.tick_params(labelsize=18)
    for j in range(len(polarisation_averages)):
        BL = "BL"
        ax1.errorbar(X1, data[i][j], yerr=errs[i][j], capsize=2, color=colours[i][j], label=f'{lb[j]} ') ## s^{{-1}}$') #\sigma$')# #s^{{-1}}$')  $^\circ s^{{-1}}$
    ax1.invert_xaxis()
    ax1.legend(frameon=True, title='$v~~\mathrm{{{BL}~\Delta t^{{-1}}}}$', title_fontsize=20,               #  $\\alpha~^\circ$   $\\nu~\mathrm{^\circ \Delta t^{-1}}$
               fontsize=18, loc=locn[i], labelspacing=0.3, handlelength=1.8, handletextpad=0.5, edgecolor='1', framealpha=0.5)
    
    if save_plots:
        
        plt.savefig(f'{new_folder_path}/{fn[i]}_CS', dpi=300, bbox_inches='tight')


#ax1.plot(X1, O_r1.reshape(11))
#ax1.plot(X1, Strip_Or_data2.reshape(11))

plt.show()
