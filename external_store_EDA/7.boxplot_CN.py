import pandas as pd
import matplotlib.pyplot as plt


store = pd.read_csv("store.csv", encoding = 'Windows-1252')
maximum = max(store["Profit"])
print(maximum)
ax = store.boxplot(by = 'State',
                   column = 'Sales',
                   figsize = (8,6))

ax.set_ylim(0,10000)

ax.set_xlabel('')
ax.set_ylabel('Sales per each State')

plt.xticks(rotation=90)
plt.suptitle('')
plt.show()