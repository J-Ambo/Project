'''
Data obtained from an online poll of 20,054 adults(18+) conducted between 20/11/15 and 2/12/15,
results weighted to be representative of all adults in the UK.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

voter_dict = {'Nothing To Lose':23, 'Global Britain':13, 'Hard-Pressed Undecideds':19,
               'Listen To DC':13, 'If It Ain\'t Broke':12, 'I\'m Alright Jacques':11, 
                'Citizens Of The World':9}

opinion_values = np.array([])
for n in np.linspace(-1,1,7):
    opinion_values = np.append(opinion_values, n)
print(opinion_values)

fig, ax = plt.subplots()
#ax.bar(opinion_values, voter_dict.values())
#ax.set_xticks(opinion_values,)

weighted_opinion = [opinion_values[i]*voter_dict[k]/100 for i, k in enumerate(voter_dict)]
mean_opinion = np.sum(weighted_opinion)
stdev_opinion = np.std(weighted_opinion)
print(mean_opinion, stdev_opinion)

