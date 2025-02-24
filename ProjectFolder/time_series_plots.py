import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil
'''Plotting the polarisation and rotation order parameters for each repetition
    as a function of time.
'''
plots_on = True
save_plots = True

data_path = r"C:\Users\44771\Desktop\Data\2002\2002_1712"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
rotation_data = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)
#print([rotation_data[0][0][:,0][i][-1] for i in range(100)])
#print(len(rotation_data[0][0][:,0]))

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]] #polarisation_data[:,0][:,1][:,-1]]
#print(polarisation_data[0][:,0][:,1])
#print(polarisation_data[:,0][:,1][:,-2])

print(ral_array)
print(rat_array)

plt.rcParams.update({"text.usetex": True, "font.family": "Times New Roman"})
time_steps = np.linspace(0, int(Timesteps), int(Timesteps))
if plots_on:
    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'
    os.makedirs(new_folder_path, exist_ok=True)
    shutil.copy(f'{data_path}/parameters.txt', new_folder_path)

    for n in range(int(Strips)):
        for i in range(int(Increments)):  #i.e. for i in range(number_of_increments)
            for r in range(int(Repetitions)):
                fig, ax = plt.subplots(figsize=(5,4))
                ax.set_xlabel('Timestep', size=22)
                ax.set_ylabel('$O_\mathrm{p/r}$', size=22)
                ax.tick_params(labelsize=18)
                ax.set_ylim(-0.05,1.05)
                ax.plot(time_steps, polarisation_data[n][i][r][0], label='Polarisation', c='blue')
                ax.plot(time_steps, rotation_data[n][i][r][0], label='Rotation', c='red')
                ax.legend(frameon=False, fontsize=18, reverse=True)
                ral = np.round(ral_array[i],1)
                rat = np.round(rat_array[i],1) + n*0.5  
                #ax.set_title(f'Rep:{r+1} Pop:{int(Population_size)} ral:{ral} rat:{rat}')

                if save_plots:
                    os.makedirs(f'{new_folder_path}/Strip{n+1}', exist_ok=True)
                    plt.savefig(f'{new_folder_path}/Strip{n+1}/I{i+1}R{r+1}.png', dpi=300, bbox_inches='tight')
                    
                plt.close()

                    
    #if save_plots:
       # shutil.copy(f'{data_path}/parameters.txt', new_folder_path)
