import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the correlation matrix from the CSV file.
# 'index_col=0' tells pandas to use the first column as row names (like R's row.names = 1)
telecom_corr_data = pd.read_csv("1.telecom.csv", index_col=0)

# --- Visualization using Seaborn ---

plt.figure(figsize=(7, 6))

sns.heatmap( #sns.heatmap(…) creates a heatmap using the data you provide.
    telecom_corr_data,  #the pandas DataFrame containing the correlation matrix.
    vmin=-1, 
    vmax=1,
    cmap=sns.diverging_palette(0, 220, as_cmap=True), #provide changing colors and smooth transition
    annot=True,        # Show correlation values 
    fmt=".3f",         # 3 decimal places
    linewidths=.5,     # lines between cells
    linecolor='white',
    cbar_kws={'label' : 'Correlation Coefficient'}
)

plt.title('Correlation Heatmap of Telecom Stocks (T, CTL, FTR, VZ, LVLT)')
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Print the loaded data
print("\n--- Correlation Matrix Data ---")
print(telecom_corr_data)
