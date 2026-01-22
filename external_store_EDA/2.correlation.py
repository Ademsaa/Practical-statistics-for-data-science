import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

store = pd.read_csv("store.csv", encoding = 'Windows-1252')

correaltion_matrix = store[["Sales","Profit", "Discount", "Quantity"]].corr(method = "pearson")
print(correaltion_matrix)
                                        
plt.figure()
sns.heatmap(correaltion_matrix,
            vmin=1,
            vmax=1,
            cmap=sns.diverging_palette(0,200, as_cmap=True),
            annot = True,
            fmt=".3f",
            linewidths=.5,
            linecolor='white',
            cbar_kws={'label': 'Correlation Coefficient'})


plt.title("Correlation hetamap of store's data")
plt.show()