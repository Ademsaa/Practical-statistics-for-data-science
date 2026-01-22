import pandas as pd
import matplotlib.pyplot as plt


delay = pd.read_csv("1.delay.csv")

#creation of boxplot 
ax = delay.transpose().plot.bar(figsize=(4,4), legend = False)
ax.set_xlabel("Cause of the delay", fontsize = 14)
ax.set_ylabel("count", fontsize = 14)

plt.show()