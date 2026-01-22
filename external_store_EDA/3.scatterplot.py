import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

store = pd.read_csv("store.csv", encoding="Windows-1252")
print(store)

plt.figure(figsize=(8,7))

# Scatter plot
plt.scatter(
    store.Sales, 
    store.Profit,
    marker ='$\u25EF$',
    s=30,
    color=(0.6, 0, 1, 0.25),
    edgecolor=None
)

# Add vertical line at x = 0 (example: AAPL mean or 0 return)
plt.axvline(x=0, color='black', linestyle='solid', linewidth=1)

# Add horizontal line at y = 0 (example: GOOG mean or 0 return)
plt.axhline(y=0, color='black', linestyle='solid', linewidth=1)

plt.title("Scatter plot Sales Profit", fontsize=12)
plt.xlabel("Sales", fontsize=12)
plt.ylabel("Profit", fontsize=12)

plt.show()

#==================================================================
