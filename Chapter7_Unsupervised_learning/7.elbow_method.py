# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
# 4. Select data from 2011 onward
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
# 6. Standardize the data
# --------------------------------------------------
scaler = StandardScaler()
top_sp_scaled = scaler.fit_transform(top_sp)

# --------------------------------------------------
# 7. Compute inertia for different numbers of clusters
# --------------------------------------------------
inertia = []

for n_clusters in range(2, 14):
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0
    )
    kmeans.fit(top_sp_scaled)
    
    # Average within-cluster sum of squares
    inertia.append(kmeans.inertia_ / n_clusters)

# Create a DataFrame for plotting
inertias = pd.DataFrame({
    'n_clusters': range(2, 14),
    'inertia': inertia
})

# --------------------------------------------------
# 8. Plot the elbow graph
# --------------------------------------------------
ax = inertias.plot(x='n_clusters', y='inertia', marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Within-Cluster Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
