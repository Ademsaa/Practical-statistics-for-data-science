import pandas as pd
import matplotlib.pyplot as plt

state = pd.read_csv("1.state.csv")

ax = (state["Population"]/1000000).plot.box()
ax.set_ylabel("Population(millions)")

plt.show()