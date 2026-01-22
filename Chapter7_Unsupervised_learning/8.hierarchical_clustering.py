# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster 

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
# 5. Transpose the data
#    Rows = stocks, columns = dates
# --------------------------------------------------
df = df.transpose()

# --------------------------------------------------
# 6. Hierarchical clustering (complete linkage)
#    No standardization – loyal to the book
# --------------------------------------------------
Z = linkage(df, method='complete')

# --------------------------------------------------
# 7. Plot dendrogram
# --------------------------------------------------
plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=df.index,
    leaf_rotation=90
)
plt.title('Hierarchical Clustering of Stocks (Complete Linkage)')
plt.ylabel('Dissimilarity')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# To extract a specific number of clusters
# --------------------------------------------------
memb = fcluster(Z, 5, criterion='maxclust')  # 4 clusters
memb = pd.Series(memb, index=df.index)

# Group stocks by cluster and print
for key, item in memb.groupby(memb):
    print(f"{key} : {', '.join(item.index)}")