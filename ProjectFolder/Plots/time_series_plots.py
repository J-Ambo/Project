import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil
'''Plotting the polarisation and rotation order parameters for each repetition
    as a function of time.
'''
plots_on = True
save_plots = False

folder_path = r'C:\Users\44771\Desktop\Data\1503\1503_2245'
for file in os.listdir(folder_path):
    print(file)

data_path = r"C:\Users\44771\Desktop\Data\1503\1503_2245\V1_56098"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
data_file_name3 = os.path.split(os.path.split(os.path.split(data_path)[0])[0])[1]
print(data_file_name3)

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
rotation_data = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)
#print(rotation_data)
parameters = {}
with open(f'{data_path}/parameters.txt', 'r') as file:
    for line in file:
        if ':' in line:
            key, value = line.strip().split(':', 1)
            parameters[key.strip()] = value.strip()

#print(parameters['Range of radius of attraction'].split('-',1)[0])

strips = int(parameters['Strips'])
increments = int(parameters['Increments'])
repetitions = int(parameters['Repetitions'])
timesteps = int(parameters['Timesteps'])

plt.rcParams.update({"text.usetex": True, "font.family": "Times New Roman"})
time_steps = np.linspace(0, timesteps, timesteps)
if plots_on:
    time_dm = time.strftime('%d%m')
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name3}/{data_file_name2}/{data_file_name1}'
    os.makedirs(new_folder_path, exist_ok=True)
    shutil.copy(f'{data_path}/parameters.txt', new_folder_path)

    for n in range(strips):
        for i in range(increments):  #i.e. for i in range(number_of_increments)
            for r in range(repetitions):
                fig, ax = plt.subplots(figsize=(5,4))
                ax.set_xlabel('Timestep', size=22)
                ax.set_ylabel('$O_\mathrm{p/r}$', size=22)
                ax.tick_params(labelsize=18)
                ax.set_ylim(-0.05,1.05)
                ax.plot(time_steps, polarisation_data[n][i][r], label='Polarisation', c='blue')
                ax.plot(time_steps, rotation_data[n][i][r], label='Rotation', c='red')
                ax.legend(frameon=False, fontsize=18, reverse=True)

                if save_plots:
                    os.makedirs(f'{new_folder_path}/Strip{n+1}', exist_ok=True)
                    plt.savefig(f'{new_folder_path}/Strip{n+1}/I{i+1}R{r+1}.png', dpi=300, bbox_inches='tight')

                plt.close()
