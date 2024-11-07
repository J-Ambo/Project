import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\44771\Downloads\Age_pop_data2015.xlsx"
#path = path.replace("\\", "/")
data_2015 = pd.read_excel(path, sheet_name='Chart data')

Ages = ['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65-89']

M_0_17 = data_2015['Male'][(data_2015['Age'] >= 0) & (data_2015['Age'] < 18)].sum()
M_18_24 = data_2015['Male'][(data_2015['Age'] >= 18) & (data_2015['Age'] < 25)].sum()
M_25_34 = data_2015['Male'][(data_2015['Age'] >= 25) & (data_2015['Age'] < 35)].sum()
M_35_44 = data_2015['Male'][(data_2015['Age'] >= 35) & (data_2015['Age'] < 45)].sum()  
M_45_54 = data_2015['Male'][(data_2015['Age'] >= 45) & (data_2015['Age'] < 55)].sum()
M_55_64 = data_2015['Male'][(data_2015['Age'] >= 55) & (data_2015['Age'] < 65)].sum()
M_65_89 = data_2015['Male'][(data_2015['Age'] >= 65) & (data_2015['Age'] < 90)].sum()

F_0_17 = data_2015['Female'][(data_2015['Age'] >= 0) & (data_2015['Age'] < 18)].sum()
F_18_24 = data_2015['Female'][(data_2015['Age'] >= 18) & (data_2015['Age'] < 25)].sum()
F_25_34 = data_2015['Female'][(data_2015['Age'] >= 25) & (data_2015['Age'] < 35)].sum()
F_35_44 = data_2015['Female'][(data_2015['Age'] >= 35) & (data_2015['Age'] < 45)].sum()  
F_45_54 = data_2015['Female'][(data_2015['Age'] >= 45) & (data_2015['Age'] < 55)].sum()
F_55_64 = data_2015['Female'][(data_2015['Age'] >= 55) & (data_2015['Age'] < 65)].sum()
F_65_89 = data_2015['Female'][(data_2015['Age'] >= 65) & (data_2015['Age'] < 90)].sum()


fig, (ax1,ax2) = plt.subplots(1,2, sharey=False)
plt.subplots_adjust(wspace=0.25)
#ax[1].set_frame_on(False)
ax1.set_yticks([])
ax1.invert_xaxis()
ax1.barh(Ages, np.asarray([M_0_17, M_18_24, M_25_34, M_35_44, M_45_54, M_55_64, M_65_89])/1000)
ax2.barh(Ages, np.asarray([F_0_17, F_18_24, F_25_34, F_35_44, F_45_54, F_55_64, F_65_89])/1000)
plt.show()