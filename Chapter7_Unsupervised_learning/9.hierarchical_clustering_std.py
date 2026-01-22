# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
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
syms1 = [
    'AAPL', 'AMZN', 'AXP', 'COP', 'COST', 'CSCO', 'CVX', 'GOOGL',
    'HD', 'INTC', 'JPM', 'MSFT', 'SLB', 'TGT', 'USB',
    'WFC', 'WMT', 'XOM'
]

# --------------------------------------------------
# 4. Select data from 2011 onward
#    Rows = dates, columns = stocks
# --------------------------------------------------
df = sp500_px.loc[
    sp500_px.index >= '2011-01-01',
    syms1
]

# --------------------------------------------------
# 5. Compute returns and clean data
# --------------------------------------------------
df = df.pct_change()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# --------------------------------------------------
# 6. Transpose data
#    Rows = stocks, columns = dates
# --------------------------------------------------
df = df.transpose()

# --------------------------------------------------
# 7. Standardize the data
#    Important for distance-based methods
# --------------------------------------------------
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# --------------------------------------------------
# 8. Hierarchical clustering
#    method='complete' → complete linkage
# --------------------------------------------------
Z = linkage(df, method='complete')

# --------------------------------------------------
# 9. Plot dendrogram
# --------------------------------------------------
plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=df.index,
    leaf_rotation=90
)
plt.title('Hierarchical Clustering of Stocks')
plt.ylabel('Dissimilarity (Distance)')
plt.tight_layout()
plt.show()
