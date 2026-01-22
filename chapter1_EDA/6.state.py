import pandas as pd
import matplotlib.pyplot as plt


state = pd.read_csv("1.state.csv")

ax = state['MurderRate'].plot.hist(density=True, 
                                   xlim=[0,12], 
                                   bins=range(1,12), 
                                   edgecolor = "black")

#this line make superposing graphs possible (plot take an optional axis ax)
state['MurderRate'].plot.density(ax=ax)
ax.set_xlabel('Murder Rate (per 100,000)')

plt.show()