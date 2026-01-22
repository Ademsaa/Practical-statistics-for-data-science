# Import necessary libraries
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the S&P 500 data
sp500_px = pd.read_csv("sp500_data.csv", index_col=0, parse_dates=True)  # replace with your file path

# Select symbols of interest
syms = sorted([
    'AAPL', 'MSFT', 'CSCO', 'INTC', 
    'CVX', 'XOM', 'SLB', 'COP', 
    'JPM', 'WFC', 'USB', 'AXP', 
    'WMT', 'TGT', 'HD', 'COST'
])

# Filter data from 2011-01-01 onward
top_sp = sp500_px.loc[sp500_px.index >= '2011-01-01', syms]

# Perform PCA on raw stock prices
sp_pca = PCA()
sp_pca.fit(top_sp)

# Create a DataFrame of explained variance
explained_variance = pd.DataFrame(sp_pca.explained_variance_, columns=['Eigenvalue'])

# Plot the scree plot for the first 10 components
ax = explained_variance.head(10).plot.bar(legend=False, figsize=(6, 4))
ax.set_xlabel('Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Scree Plot of Top 10 Principal Components')

# Set x-axis labels as "Component 1", "Component 2", ...
ax.set_xticklabels([f'Component {i}' for i in range(1, 11)], rotation=45)

plt.show()
