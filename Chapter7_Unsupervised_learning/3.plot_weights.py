# Import necessary libraries
import pandas as pd
import numpy as np
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

# Perform PCA on raw stock prices (book approach)
sp_pca = PCA()
sp_pca.fit(top_sp)

# Extract loadings for the top 5 principal components
loadings = pd.DataFrame(sp_pca.components_[0:5, :], columns=top_sp.columns)

# Determine maximum absolute loading for consistent y-axis
maxPC = 1.01 * np.max(np.abs(loadings.values))

# Create subplots: 5 PCs, stacked vertically
fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

# Loop through each PC and plot its loadings
for i, ax in enumerate(axes):
    pc_loadings = loadings.loc[i, :]
    # Color bars based on sign: positive=blue, negative=red
    colors = ['C0' if l > 0 else 'C1' for l in pc_loadings]
    
    # Horizontal line at y=0
    ax.axhline(0, color='#888888', linewidth=1)
    
    # Bar plot of loadings
    pc_loadings.plot(kind='bar', ax=ax, color=colors)
    
    # Label the y-axis with PC number
    ax.set_ylabel(f'PC{i+1}', fontsize=10)
    
    # Set consistent y-axis limits
    ax.set_ylim(-maxPC, maxPC)

# Set x-axis label only on the bottom subplot
axes[-1].set_xlabel('Stock Symbols', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()
