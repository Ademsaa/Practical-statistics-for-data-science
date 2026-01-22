import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

store = pd.read_csv("store.csv", encoding = 'Windows-1252')

filtred_store = store.loc[(store.Sales <= 1000) &
                          (store.Profit < 800) ].copy()

states_name = ['Alabama', 'Iowa', 'New Jersey', 'Vermont']

states = filtred_store.loc[filtred_store.State.isin(states_name),:].copy()


def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette('blue', as_cmap = True)
    plt.hexbin(x,y, gridsize=25, cmap=cmap, **kwargs)

g = sns.FacetGrid(states, col= 'State', col_wrap = 2, height = 4, aspect=1.2)
g.map(hexbin, 'Sales', 'Profit',
      extent = [0, 150, 0, 150      ])

g.set_axis_labels('Sales', 'Profit')
g.set_titles('state {col_name}')


for ax in g.axes.flat:
    ax.grid(True, linestyle = '--', alpha=0.5, color='gray')

plt.tight_layout()
plt.show()
