import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

data_path = r"C:\Users\44771\Desktop\Data\2601\2601_2225"
#data_path = r"C:\Users\44771\Desktop\Data\2601\2601_1352"
#data_path = r"C:\Users\44771\Desktop\Data\1701\1701_1022"
#data_path = r"C:\Users\44771\Desktop\Data\2701\2701_1435"

save_plots = False

data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

rotation_order_parameters = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)
polarisation_order_parameters = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)

rotation = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)
polarisation = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)

bins = np.linspace(0, 1, 30)
R_plain_data = rotation[0][0][:,0]
R_focus_data  = [R_plain_data[i][-1] for i in range(len(R_plain_data))]#[np.mean(R_plain_data[i][-1:]) for i in range(len(R_plain_data))]#
R_inds = np.digitize(R_focus_data, bins)
#print(R_inds)

P_plain_data = polarisation[0][0][:,0]
P_focus_data  = [P_plain_data[i][-1] for i in range(len(P_plain_data))]#[np.mean(P_plain_data[i][-1:]) for i in range(len(P_plain_data))]#
P_inds = np.digitize(P_focus_data, bins)


histogram_bins = np.linspace(0, 1, 20) 
H, xedges, yedges = np.histogram2d(R_focus_data, P_focus_data, bins=(histogram_bins, histogram_bins))
fig, ax = plt.subplots()
X, Y = np.meshgrid(xedges, yedges)
pcm = ax.pcolormesh(X, Y, H.T, shading='auto', cmap='gnuplot')
fig.colorbar(pcm, ax=ax, label='Frequency')

ax.set_xlabel('Rotation')
ax.set_ylabel('Polarisation')
ax.set_title('Frequency Plot')



if save_plots:
    new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'
    plt.savefig(f'{new_folder_path}/Frequency.png', dpi=300, bbox_inches='tight')

plt.show()


