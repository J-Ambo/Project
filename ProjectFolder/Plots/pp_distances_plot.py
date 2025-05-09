import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil

#data_path = r"C:\Users\44771\Desktop\Data\1702\1702_1051"
folder_path = r"C:\Users\44771\Desktop\Data\2703\2703_1023"
for file in os.listdir(folder_path):

    data_path = f"{folder_path}/{file}"
    print(data_path)
    data_file_name1 = os.path.split(data_path)[1]
    data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
    data_file_name3 = os.path.split(os.path.split(os.path.split(data_path)[0])[0])[1]

    #min_distances_data = np.load(f'{data_path}/predator_prey_distances.npy', allow_pickle=True)
    #attack_number_data = np.load(f'{data_path}/predator_attack_number.npy', allow_pickle=True)
    density_data = np.load(f'{data_path}/density_data.npy', allow_pickle=True)
   # target_density = np.load(f'{data_path}/target_density_data.npy', allow_pickle=True)
   # predator_success = np.load(f'{data_path}/predator_success.npy', allow_pickle=True)

    parameters = {}
    with open(f'{data_path}/parameters.txt', 'r') as file:
        for line in file:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                parameters[key.strip()] = value.strip()

    strips = int(parameters['Strips'])
    increments = int(parameters['Increments'])
    repetitions = int(parameters['Repetitions'])
    timesteps = int(parameters['Timesteps'])

   # N_strips = len(min_distances_data)
   # N_reps = len(min_distances_data[0][0])

    s = 0
    i = 0
    r = 0

   # print(np.nanmin(predator_success[s][i], axis=1))
    fig, ax = plt.subplots()
   # ax.set_ylim(0, 0.02)


    for s in range(strips):
       # ax.plot(np.arange(0,repetitions), np.nanmin(predator_success[s][i], axis=1))
        for r in range(repetitions):
            mean = np.nanmean(density_data[s][i][r], axis=1)
            errs = np.nanstd(density_data[s][i][r], axis=1) / np.sqrt(len(density_data[s][i][r][0]))
        # print(density_data.shape, density_data[0][0][0][0][[1,2]])
            y1 = mean - errs
            y2 = mean + errs
            plt.fill_between(np.linspace(0,len(density_data[s][i][r]), len(density_data[s][i][r])), y1, y2, alpha=0.4)
            plt.plot(np.linspace(0,len(density_data[s][i][r]), len(density_data[s][i][r])), mean)
            #plt.plot(np.linspace(0,len(density_data[s][i][r]), len(density_data[s][i][r])), target_density[s][r][r], color='r')

            #ax.plot(np.linspace(0,len(predator_success[s][i][r]), len(predator_success[s][i][r])), predator_success[s][i][r])

            


plt.show()


print(np.mean(attack_number_data[0][0][:,-1]))
N_strips = len(min_distances_data)
N_reps = len(min_distances_data[0][0])

minimums = np.nanmin(min_distances_data, axis=3)
averages = np.mean(minimums, axis=2).reshape(N_strips)
errors = np.std(minimums, axis=2).reshape(N_strips) / np.sqrt(N_reps)
#print('Distance',min_distances_data[0][0])
#print('MIN',minimums)
print(averages)
print(errors)

#plt.errorbar(np.linspace(0,1, N_strips), averages, errors, fmt='o')
#plt.show()


data = min_distances_data[s][i][r]
#plt.plot(np.linspace(1, len(data), len(data)), data)
#plt.show()

for i in range(N_strips):
    #data = min_distances_data[0][0][i]
    for j in range(N_reps):
        data = min_distances_data[i][0][j]
        plt.plot(np.linspace(1, len(data), len(data)), data)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Minimum separation', fontsize = 14)
#plt.show()
