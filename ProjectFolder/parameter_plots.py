import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import time
import os
import shutil
from matplotlib.ticker import FormatStrFormatter
'''Plotting the polarisation and rotation order parameters as functions of
    the alignment and attraction zone widths.
'''
plots_on = True
save_plots = False


data_path = r"C:\Users\44771\Desktop\Data\2912\2912_1547"
data_file_name1 = os.path.split(data_path)[1]
data_file_name2 = os.path.split(os.path.split(data_path)[0])[1]

polarisation_data = np.load(f'{data_path}/polarisation_data.npy', allow_pickle=True)
polarisation_averages = np.load(f'{data_path}/polarisation_averages.npy', allow_pickle=True)
rotation_averages = np.load(f'{data_path}/rotation_averages.npy', allow_pickle=True)

polarisation_errors = np.load(f'{data_path}/polarisation_errors.npy', allow_pickle=True)
rotation_errors = np.load(f'{data_path}/rotation_errors.npy', allow_pickle=True)

Population_size, Arena_radius, Timesteps, Repetitions, Increments, Strips = (polarisation_data[0][0][0][1][0], polarisation_data[0][0][0][1][1],
                                                                polarisation_data[0][0][0][1][2], polarisation_data[0][0][0][1][3], 
                                                                polarisation_data[0][0][0][1][4], polarisation_data[0][0][0][1][5])

ral_array = [sub_array[-2] for sub_array in polarisation_data[0][:,0][:,1]]
rat_array = [sub_array[-1] for sub_array in polarisation_data[0][:,0][:,1]]

x1 = np.asarray(ral_array)
y1 = np.asarray(rat_array) 
X1 = x1 - 1 
Y1 = y1 - x1[0]
O_r1 = rotation_averages
O_p1 = polarisation_averages
print(X1)
print(Y1)

O_i1 = [O_p1, O_r1]
label = ['$O_p$', '$O_r$']
colour = [plt.get_cmap('Blues')(0.7), plt.get_cmap('Reds')(0.8)]
Errs = [polarisation_errors, rotation_errors]

print(max(Errs[0].flatten()), max(Errs[1].flatten()))
print(min(Errs[0].flatten()), min(Errs[1].flatten()))
#time_dm = time.strftime('%d%m')
new_folder_path = f'C:/Users/44771/Desktop/Plots/{data_file_name2}/{data_file_name1}'

plt.rcParams.update({"text.usetex": True, "font.family": "Nimbus Roman"})
for i in range(2):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 5))
    fig.tight_layout(rect=(-0.03,0.05,1,1.02))
    ax.plot_surface(X1, np.tile(Y1, (31,1)).T, O_i1[i], rstride=1, cstride=1, edgecolor=colour[i], facecolor=colour[i], alpha=0.2)
    cont = ax.contourf(X1, Y1, Errs[i], zdir='z', offset=-0.2, cmap='gnuplot', levels=20, alpha =0.8)
    cbar = fig.colorbar(cont, location='left', shrink=0.5, pad=0.01, format=FormatStrFormatter('%.2f'))
    cbar.set_ticks([min(Errs[i].flatten()),max(Errs[i].flatten())/2,max(Errs[i].flatten())])
    cbar.ax.tick_params(labelsize=14)
    cont.set_clim([min(Errs[i].flatten()),max(Errs[i].flatten())])
    ax.set_xlabel('$\Delta r_{al}$', size=16)
    ax.set_ylabel('$\Delta r_{at}$', size=16)
    ax.set_zlabel(label[i], size=16, labelpad=0.4)
    ax.set_zlim(-0.2, 1)
    ax.set_ylim(0.0, Y1[-1])
    ax.set_xlim(0.0, X1[-1])
    ax.tick_params(labelsize=14)
    ax.invert_xaxis()

    ax.xaxis.set_major_locator(ticker.FixedLocator([0, 2, 4, 6, 8, 10, 12, 14]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 2, 4, 6, 8, 10, 12, 14]))
    ax.zaxis.set_major_locator(ticker.FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    ax.tick_params(pad=1)

    ax.view_init(elev=25, azim=-50)
    #plt.savefig(f"C:/Users/44771/Desktop/GifImages/pan/{label[i]}_rot_{elev}_{azim}.png")

    if save_plots:
        os.makedirs(new_folder_path, exist_ok=True)#
        #plt.savefig(f'{plot_dir}/{label[i]}_{data_file_name2}-{data_file_name3}-{data_file_name1}', dpi=300, bbox_inches=None)#
        plt.savefig(f'{new_folder_path}/{label[i]}', dpi=300, bbox_inches=None)


'''for i in range(2):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_ylim(0.0, Y1[-1])
    ax.set_xlim(0.0, X1[-1])
    ax.contourf(X1, Y1, Errs[i], cmap='gray_r', levels=20)
    ax.invert_xaxis()
    if save_plots:
        plt.savefig(f'{new_folder_path}/Errs_{label[i]}', dpi=300, bbox_inches=None)'''
plt.show()