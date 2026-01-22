import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


store = pd.read_csv("store.csv", encoding="Windows-1252")

plt.figure(figsize=(8,7))
sns.kdeplot(
x = store.Sales,
y = store.Profit,
#levels = 10,
color = "white",
linewidth = 1.2
)

plt.title("Contour plot Sales Profit", fontsize=12)
plt.xlabel("Sales", fontsize=12)
plt.ylabel("Profit", fontsize=12)

plt.show()