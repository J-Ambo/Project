import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

data_path = r"C:\Users\44771\Desktop\Data\1701\1701_1022"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

rotation_order_parameters = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)
polarisation_order_parameters = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)

