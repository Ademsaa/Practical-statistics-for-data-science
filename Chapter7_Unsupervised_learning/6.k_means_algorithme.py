# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --------------------------------------------------
# 2. Load S&P 500 data
# --------------------------------------------------
sp500_px = pd.read_csv(
    "sp500_data.csv",
    index_col=0,
    parse_dates=True
)

# --------------------------------------------------
# 3. Define stock symbols
# --------------------------------------------------
syms = sorted([
    'AAPL', 'MSFT', 'CSCO', 'INTC',
    'CVX', 'XOM', 'SLB', 'COP',
    'JPM', 'WFC', 'USB', 'AXP',
    'WMT', 'TGT', 'HD', 'COST'
])

# --------------------------------------------------
# 4. Select prices from 2011 onward
# --------------------------------------------------
top_sp = sp500_px.loc[
    sp500_px.index >= '2011-01-01',
    syms
]

# --------------------------------------------------
# 5. Compute returns and clean data
# --------------------------------------------------
top_sp = top_sp.pct_change()
top_sp.replace([np.inf, -np.inf], np.nan, inplace=True)
top_sp.dropna(inplace=True)

# --------------------------------------------------
# 6. Standardize the data (important for K-means)
# --------------------------------------------------
scaler = StandardScaler()
top_sp_scaled = scaler.fit_transform(top_sp)

# --------------------------------------------------
# 7. Apply K-means clustering
# --------------------------------------------------
kmeans = KMeans(
    n_clusters=5,
    n_init=10,
    max_iter=300,
    random_state=42
)
kmeans.fit(top_sp_scaled)

# --------------------------------------------------
# 8. Access results
# --------------------------------------------------
labels = kmeans.labels_
centers = pd.DataFrame(kmeans.cluster_centers_, columns=syms)

# Optional: check cluster sizes
from collections import Counter
print("Cluster sizes:", Counter(labels))

# --------------------------------------------------
# 9. Visualize cluster centers as bar plots
# --------------------------------------------------
f, axes = plt.subplots(nrows=5, ncols=1, figsize=(8, 10), sharex=True)

for i, ax in enumerate(axes):
    # Select the center of cluster i
    center = centers.loc[i, :]
    
    # Maximum absolute value for symmetric y-axis
    maxPC = 1.01 * np.max(np.abs(center))
    
    # Color positive returns in C0, negative in C1
    colors = ['C0' if val > 0 else 'C1' for val in center]
    
    # Horizontal line at 0
    ax.axhline(color='#888888')
    
    # Plot bar chart
    center.plot.bar(ax=ax, color=colors)
    
    # Label cluster
    ax.set_ylabel(f'Cluster {i + 1}')
    
    # Set symmetric y-axis
    ax.set_ylim(-maxPC, maxPC)

plt.tight_layout()
plt.show()
