import pandas as pd
import matplotlib.pyplot as plt

store = pd.read_csv("store.csv", encoding = "Windows-1252")

store_reshaped = store.loc[(store.Sales < 200) &
                           (store.Sales > 0) &
                            (store.Profit < 50)&
                            (store.Profit > -10),:]

ax = store_reshaped.plot.hexbin(x = 'Sales',
                                y = 'Profit',
                                gridsize = 30,
                                sharex = False,
                                figsize = (8,7),
                                color = "green")

ax.set_xlabel('Sales ($)')
ax.set_ylabel('Profit ($)')

plt.show()