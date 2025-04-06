import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import os
import shutil
import matplotlib.colors as colors

save_plots = True
combined_plot = True

plt.rcParams.update({"text.usetex": True, "font.family": "Nimbus Roman"})

folderPath = r'C:\Users\44771\Desktop\Data\HeatMapData\0304_1521_gd30'
data_file_name1 = os.path.split(folderPath)[1]
data_file_name2 = os.path.split(os.path.split(folderPath)[0])[1]

PopSize=None
def load_data(folder_path):
    datasets = []

    for file in os.listdir(folder_path):

        data_path = f"{folder_path}/{file}"
     #   print(data_path)
        parameters = {}
        with open(f'{data_path}/parameters.txt', 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    parameters[key.strip()] = value.strip()
        global PopSize
        PopSize = int(parameters['Population'])
        
        rotation_data = np.load(f'{data_path}/rotation_data.npy', allow_pickle=True)
        polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)

        N_strips = len(rotation_data)
        N_reps = len(rotation_data[0][0])
        
        for r in range(N_reps):
           # R_data = rotation_data[0][0][r]
           # P_data = polarisation_data[0][0][r]
            R_focus_data = np.array(rotation_data[0][0][r][-1000:]) #[rotation_data[0][0][r][-i-1] for i in range(50)]
            P_focus_data = np.array(polarisation_data[0][0][r][-1000:])    #[polarisation_data[0][0][r][-i-1] for i in range(50)]
            datasets.append((R_focus_data, P_focus_data))
    return datasets

def combined_frequency_plot(*datasets, axis, bins, threshold, cmap_name='jet'):
    histogram_bins = np.linspace(0, 1, bins)
    combined_histogram = None
    
    for (R_data, P_data) in datasets:
        #print(R_data)
        H, xedges, yedges = np.histogram2d(R_data, P_data, bins=(histogram_bins, histogram_bins))
        #H_masked = np.ma.masked_less(H, threshold)

        if combined_histogram is None:
            combined_histogram = H  #H_masked
        else:
            combined_histogram += H  #H_masked

    combined_histogram_masked = np.ma.masked_less(combined_histogram, threshold)

  #  fig, ax = plt.subplots(figsize=(6.2,5))  #()
  #  axis.text(0.6, 0.8, f'{PopSize} FISH', size=20, color='white', horizontalalignment='left', transform=axis.transAxes)
    X, Y = np.meshgrid(xedges, yedges)

  #  colourmap = plt.get_cmap(cmap_name)#(np.linspace(0.02, 1, 256))
  #  colourmap.set_under(color='black')# colourmap.set_bad(color='black')
   # im = plt.imshow(2*combined_histogram_masked.T, origin='lower',extent=[0, 1, 0, 1],
   #      cmap=colourmap, norm=colors.Normalize(vmin=1, vmax=combined_histogram.max()), interpolation = "bilinear")
  
 #   cbar = fig.colorbar(im, spacing='proportional', shrink=0.9, ax=ax)
 #   cbar.set_ticks([0, combined_histogram.max()])
 #   cbar.set_ticklabels(['$\mathrm{p_{min}}$', '$\mathrm{p_{max}}$'], size=26)

   ## pcm = ax.pcolormesh(X, Y, 3*combined_histogram_masked.T, cmap=cmap, shading='auto')
   ## pcm.set_clim(0, combined_histogram.max())
   ## cbar = fig.colorbar(pcm, ax=ax)
   ## cbar.ax.tick_params(labelsize=22)  
    #cbar.set_label('Frequency', size=22) 
   ## cbar.set_ticks([0, combined_histogram.max()])
   ## cbar.set_ticklabels(['$\mathrm{p_{min}}$', '$\mathrm{p_{max}}$'], size=26)

   # ax.set_xlabel('$O_\mathrm{r}$', size=32)
   # ax.set_ylabel('$O_\mathrm{p}$', size=32)
   
    # ax.tick_params(labelsize=24)
  #  ax.set_yticks([])# ax.set_yticks([0,0.5,1])
   # ax.set_xticks([0,0.5,1])

    if save_plots:
        
        new_folder_path = f'Plot_PNGs/{data_file_name2}/{data_file_name1}'
        os.makedirs(new_folder_path, exist_ok=True)
        plt.savefig(f'{new_folder_path}/Frequency2.png', dpi=300, bbox_inches='tight')
    
    return combined_histogram_masked
#datasets = load_data(folderPath)
#combined_frequency_plot(*datasets, bins=50, threshold=0)


HeatMapPath = r'C:\Users\44771\Desktop\Data\HeatMapData'
All_folders = [f'{HeatMapPath}/0304_1521_gd30', f'{HeatMapPath}/0404_0708_gd70', f'{HeatMapPath}/0304_1812_gd150', f'{HeatMapPath}/0204_2134_gd300']
Fig, Ax = plt.subplots(2,2, figsize=(7,7))

Fig.supylabel('Polarisation $O_\mathrm{p}$', x=0, y=0.48, fontsize='22')
Fig.supxlabel('Rotation $O_\mathrm{r}$', x=0.48, y=0, fontsize='22')
colourmap = 'jet'

x_035 = np.linspace(0,0.35)
y_035 = np.linspace(0,0.35)

clrmap = plt.get_cmap(colourmap).copy()
for i, ax in enumerate(Ax.flat):
    #print(f"Axis index: {i}")
    ax.set_facecolor('k')
    ax.tick_params(length=0)
   # ax.set_aspect('equal')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1g}'.format(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1g}'.format(y)))
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=18)
    ax.text(0.26, 0.68, '$\mathbf{P}$', color='white', size=14)
    ax.text(0.92, 0.26, '$\mathbf{M}$', color='white', size=14)
    ax.text(0.26, 0.26, '$\mathbf{S}$', color='white', size=14)
    ax.text(0.45, 0.45, '$\mathbf{T}$', color='white', size=14)

    print(All_folders[i])
    DataSet = load_data(All_folders[i])
    MapData = combined_frequency_plot(*DataSet, axis=ax, bins=50, threshold=0, cmap_name='jet')
    clrmap.set_under('black')
    im = ax.imshow(2.3*MapData.T, origin='lower', extent=[0, 1, 0, 1],
         cmap=clrmap, vmin=1, vmax=MapData.max(), interpolation="bilinear")
    line1 = ax.hlines([0.35, 0.65], 0, 0.35, 'r', linestyle=(0, (6, 5)), linewidth=2, alpha=0.8)  
    line2 = ax.hlines(0.35, 0.65, 1, 'r', linestyle=(0, (6, 5)), linewidth=2, alpha=0.8)  

    line3 = ax.vlines([0.35, 0.65], 0, 0.35, 'r', linestyle=(0, (6, 5)), linewidth=2,  alpha=0.8)
    line4 = ax.vlines(0.35, 1, 0.65, 'r', linestyle=(0, (6, 5)), linewidth=2,  alpha=0.8)

cbar_ax = Fig.add_axes([0.88, 0.25, 0.04, 0.4])
cbar = Fig.colorbar(im, cax=cbar_ax)        #cax=cbar_ax)
#cbar = Fig.colorbar(im, ax=Ax.ravel().tolist(), spacing='proportional', shrink=0.8)
cbar.set_ticks([])#[0, MapData.max()])
#cbar.set_ticklabels(['$\mathrm{p_{min}}$', '$\mathrm{p_{max}}$'], size=26)
Fig.subplots_adjust(wspace=0.06, hspace=0.06, right=0.85, left=0.1, bottom=0.08, top=0.84)
Fig.axes[4].set_title('$\mathrm{p_{max}}$', fontsize=22)
Fig.axes[4].set_xlabel('$\mathrm{p_{min}}$', fontsize=22)

Ax[0, 0].set_xticks([])
Ax[0, 0].set_yticks([0,0.5,1])
Ax[0,0].text(0.6, 0.85, '30 FISH', size=20, color='white')

Ax[0, 1].set_xticks([])
Ax[0, 1].set_yticks([])
Ax[0,1].text(0.6, 0.85, '70 FISH', size=20, color='white')

Ax[1, 0].set_xticks([0,0.5,1])
Ax[1, 0].set_yticks([0,0.5,1])
Ax[1,0].text(0.55, 0.85, '150 FISH', size=20, color='white')

Ax[1, 1].set_xticks([0,0.5,1])
Ax[1, 1].set_yticks([])
Ax[1,1].text(0.55, 0.85, '300 FISH', size=20, color='white')
Ax[1,1].text(0.26, 0.68, '$\mathbf{P}$', color='black', size=14)

if save_plots:
        
   # new_folder_path = f'HeatMapRath/{data_file_name2}/{data_file_name1}'
   # os.makedirs(new_folder_path, exist_ok=True)
    plt.savefig(f'{HeatMapPath}/Frequency.png', dpi=300)  # bbox_inches='tight')


plt.show()



