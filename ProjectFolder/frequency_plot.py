import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import shutil

#data_path = r"C:\Users\44771\Desktop\Data\2601\2601_2225"
#data_path = r"C:\Users\44771\Desktop\Data\2601\2601_1352"
#data_path = r"C:\Users\44771\Desktop\Data\1701\1701_1022"
#data_path2 = r"C:\Users\44771\Desktop\Data\2701\2701_1435"
#data_path = r"C:\Users\44771\Desktop\Data\3101\3101_1523"

#d1 = r"C:\Users\44771\Desktop\Data\2801\2801_0137"
#d2 = r"C:\Users\44771\Desktop\Data\2801\2801_0708"
#d3 = r"C:\Users\44771\Desktop\Data\2801\2801_1414"
#d4 = r"C:\Users\44771\Desktop\Data\2801\2801_1846"
#d5 = r"C:\Users\44771\Desktop\Data\2801\2801_2015"

#d1 = r"C:\Users\44771\Desktop\Data\2202\2202_1436"
d1 = r"C:\Users\44771\Desktop\Data\2602\2602_0435"
data_file_name1 = os.path.split(d1)[1]
data_file_name2 = os.path.split(os.path.split(d1)[0])[1]

#d2 = r"C:\Users\44771\Desktop\Data\2202\2202_1436"
#d3 = r"C:\Users\44771\Desktop\Data\2202\2202_1436"
#d4 = r"C:\Users\44771\Desktop\Data\2202\2202_1436"
#d5 = r"C:\Users\44771\Desktop\Data\2202\2202_1436"

save_plots = True
combined_plot = True

plt.rcParams.update({"text.usetex": True, "font.family": "Nimbus Roman"})

def load_data(data_path):
    datasets = []
    rotation_data = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)
    polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)

    N_strips = len(rotation_data)
    N_reps = len(rotation_data[0][0])
    print(N_reps)
    for data in range(N_strips):
        R_data = rotation_data[data][0][:,0]
        P_data = polarisation_data[data][0][:,0]
        R_focus_data = [R_data[i][-1] for i in range(N_reps)]
        P_focus_data = [P_data[i][-1] for i in range(N_reps)]
        datasets.append((R_focus_data, P_focus_data))

    '''datasets = []
    for path in data_paths:
        rotation = np.load(f'{path}/rotation_data.npy', allow_pickle=True)
        polarisation = np.load(f'{path}/polarisation_data.npy', allow_pickle=True)
        R_plain_data = rotation[0][0][:,0]
        P_plain_data = polarisation[0][0][:,0]
        R_focus_data = [R_plain_data[i][-1] for i in range(len(R_plain_data))]
        P_focus_data = [P_plain_data[i][-1] for i in range(len(P_plain_data))]
        datasets.append((R_focus_data, P_focus_data))'''
    return datasets

def combined_frequency_plot(*datasets, bins, threshold, cmap_name='turbo'):
    #print(datasets)
    histogram_bins = np.linspace(0, 1, bins)
    combined_histogram = None

    for (R_data, P_data) in datasets:
        H, xedges, yedges = np.histogram2d(R_data, P_data, bins=(histogram_bins, histogram_bins))
        H_masked = np.ma.masked_less(H, threshold)
        
        if combined_histogram is None:
            combined_histogram = H  #H_masked
        else:
            combined_histogram += H  #H_masked
        #print(combined_histogram)
   # print(combined_histogram)

    combined_histogram_masked = np.ma.masked_less(combined_histogram, threshold)
   # print(combined_histogram_masked)
    
    '''combined_histogram_masked[1][-2] = 139
    combined_histogram_masked[2][-2] = 145
    combined_histogram_masked[3][-2] = 121
    combined_histogram_masked[4][-2] = 3
    combined_histogram_masked[1][-3] = 112
    combined_histogram_masked[2][-3] = 96
    combined_histogram_masked[4][-3] = 11
    combined_histogram_masked[3][-4] = 8'''

  #  print(combined_histogram_masked[0:4][:,-2])

    fig, ax = plt.subplots(figsize=(6.2,5))  #()
    X, Y = np.meshgrid(xedges, yedges)

    cmap = plt.get_cmap(cmap_name)(np.linspace(0.02, 1, 256))
   # cmap = mpl.colormaps[cmap_name].copy()

    cmap = mpl.colors.ListedColormap(cmap)
    cmap.set_bad(color='black')
    pcm = ax.pcolormesh(X, Y, 30*combined_histogram_masked.T, shading='auto', cmap=cmap)
    pcm.set_clim(0, combined_histogram.max())
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=22)  
    #cbar.set_label('Frequency', size=22) 
    cbar.set_ticks([0, combined_histogram.max()])
    cbar.set_ticklabels(['$\mathrm{p_{min}}$', '$\mathrm{p_{max}}$'], size=26)

    ax.set_xlabel('$O_\mathrm{r}$', size=32)
   # ax.set_ylabel('$O_\mathrm{p}$', size=32)
    ax.annotate('70 FISH', xy=(0,0), xytext=(0.6, 0.8), size=28, color='white')
    ax.tick_params(labelsize=24)
    ax.set_yticks([])# ax.set_yticks([0,0.5,1])
    ax.set_xticks([0,0.5,1])

    if save_plots:
        
        new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'
        os.makedirs(new_folder_path, exist_ok=True)
        plt.savefig(f'{new_folder_path}/Frequency_1.png', dpi=300, bbox_inches='tight')

    plt.show()  

datasets = load_data(d1)
combined_frequency_plot(*datasets, bins=30, threshold=5)

