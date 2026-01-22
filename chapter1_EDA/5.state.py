import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the CSV file
state = pd.read_csv("1.state.csv")

# 2. This line calculates the boundary points (breaks) to 11 points = 10 bars
breaks = np.linspace(
    start=state["Population"].min(),
    stop=state["Population"].max(),
    num=11
)
print(breaks, "\n")

# 3. This line creates the bins by passing the breaks (boundary points)
# right=True is the default in Pandas (right value is included)
# include_lowest=True ensures the minimum value of the first bin is included
pop_freq = pd.cut(
    state["Population"], 
    bins=breaks, 
    right=True, 
    include_lowest=True
)

# 4. This line generates the frequency table 
# We use .sort_index() to ensure the bins are ordered from lowest to highest.
frequency_table = pop_freq.value_counts().sort_index()
print(frequency_table)


#creating histogram and customization using matplotlib library
#using plot.hist panda methods draw direclty the histogram
ax = (state["Population"]/1000000).plot.hist(figsize=(6,6), bins = 10, edgecolor="black", linewidth=0.7)
ax.set_xlim(left=0)#to make histogram starts from 0


ax.set_title("Distribution of population over states", fontsize = 14)
ax.set_xlabel("Population(millions)", fontsize = 12)
ax.set_ylabel("Count or Frequency", fontsize = 12)
ax.tick_params(labelsize = 10, color = "red")


plt.show()