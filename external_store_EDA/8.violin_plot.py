import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

store = pd.read_csv("store.csv", encoding = 'Windows-1252')

plt.figure(figsize = (10,6))

ax = sns.violinplot(
    data = store,
    x = 'State',
    y = 'Sales',
    inner = 'quartile',
    color = 'skyblue',
)

ax.set_ylim(0,10000)

ax.set_xlabel('')
ax.set_ylabel('Sales')
ax.set_title('Sales per state')


plt.xticks(rotation = 90)
plt.show()
