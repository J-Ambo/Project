import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil

data_path = r"C:\Users\44771\Desktop\Data\1002\1002_1056"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

min_distances_data = np.load(f'{data_path}/predator_prey_distances.npy', allow_pickle=True)
print(len(min_distances_data[0][0][7]))
s = 0
i = 0
r = 4 
data = min_distances_data[s][i][r]
plt.plot(np.linspace(1, len(data), len(data)), data)
plt.show()

for i in range(len(min_distances_data[0][0])):
    data = min_distances_data[0][0][i]
    #plt.plot(np.linspace(1, len(data), len(data)), data)
plt.xlabel('Time', fontsize = 14)
plt.ylabel('Minimum separation', fontsize = 14)
#plt.show()
