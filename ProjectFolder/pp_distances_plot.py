import numpy as np
from matplotlib import pyplot as plt
import time
import os
import shutil

data_path = r"C:\Users\44771\Desktop\Data\0902\0902_1419"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

min_distances_data = np.load(f'{data_path}/predator_prey_distances.npy', allow_pickle=True)
print(min_distances_data)
#plt.plot(min_distances_data, np.linspace(1, len(min_distances_data)-1))
