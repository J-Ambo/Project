from matplotlib import pyplot as plt
import numpy as np
import os
import shutil

folder_path = r'C:\Users\44771\Desktop\Data\0604\0604_1952'
save_plots = False

fig, ax = plt.subplots(figsize=(6,5))
fig2, ax2 = plt.subplots(figsize=(6,5))
for file in os.listdir(folder_path):
    data_path = f"{folder_path}/{file}"
    data_file_name1 = os.path.split(data_path)[1]
    data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]
    data_file_name3 = os.path.split(os.path.split(os.path.split(data_path)[0])[0])[1]

    predator_prey_angles = np.load(f'{data_path}/predator_prey_angles.npy', allow_pickle=True)
    prey_orientation = np.load(f'{data_path}/prey_orientation.npy', allow_pickle=True)
    mean_distance2prey = np.load(f'{data_path}/mean_distance2prey.npy', allow_pickle=True)

   # print(np.trim_zeros(np.mean(predator_prey_angles, axis=2)[0]))
   # print(np.trim_zeros(np.mean(prey_orientation, axis=2)[0]))
   # print(np.trim_zeros(mean_distance2prey[0]))    

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

    all_angles = np.array([row for row in predator_prey_angles[0] if not np.all(row == 0)])
    all_orientations = np.array([row for row in prey_orientation[0] if not np.all(row == 0)])
    print(all_angles[0], all_angles[0][all_angles[0] <=10], all_orientations[0][all_angles[0] <=10])
   # print(all_orientations)

    previous_theta = 0
    average_theta2s = []
    for theta in np.arange(10,180,10):   
        theta_2s = []
       # print(previous_theta, theta)
        for t in range(len(all_orientations)):
            mask = (all_angles[t] > previous_theta) & (all_angles[t] <= theta) 
      #      print(all_orientations[t][mask])
            theta_2s.extend(all_orientations[t][mask])
        previous_theta = theta

     #   print(np.nanmean(theta_2s))
        average_theta2s.append(np.nanmean(theta_2s))
    print(average_theta2s)

    ax.scatter(np.arange(10,180,10), average_theta2s)

    plt.rcParams.update({"text.usetex": True, "font.family": "Times New Roman"})

    for r in range(repetitions):
        theta_one = np.trim_zeros(np.mean(predator_prey_angles, axis=2)[r])
        theta_two = np.trim_zeros(np.mean(prey_orientation, axis=2)[r])
        
        ax.set_xlabel('$\\theta_1$', size=22)
        ax.set_ylabel('$\\theta_2$', size=22)
        ax.tick_params(labelsize=18)
        ax.set_xlim(0,180)
        ax.set_ylim(0,180)
        ax.set_xticks([0,45,90,135,180])
        ax.set_yticks([0,45,90,135,180])
        ax.plot(theta_one, theta_two)

        mean_distances = np.trim_zeros(mean_distance2prey[r])
        ax2.plot(theta_one, mean_distances)
      #  ax2.scatter(theta_two, mean_distances)
        ax2.set_xlim(0,180)
        ax2.set_xticks([0,45,90,135,180])
        ax2.set_xlabel('$\\theta_1$', size=22)
        ax2.set_ylabel('Mean distance, BL', size=22)
        

    if save_plots:
        new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name3}/{data_file_name2}/{data_file_name1}'
        os.makedirs(new_folder_path, exist_ok=True)
        shutil.copy(f'{data_path}/parameters.txt', new_folder_path)

        os.makedirs(f'{new_folder_path}/Strip{n+1}', exist_ok=True)
        plt.savefig(f'{new_folder_path}/Strip{n+1}/I{i+1}R{r+1}.png', dpi=300, bbox_inches='tight')

plt.show()