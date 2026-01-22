# -----------------------------
# Import required libraries
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


sp500_px = pd.read_csv("sp500_data.csv", index_col=0, parse_dates=True)
# -----------------------------
# Define the stock symbols
# -----------------------------
syms = [
    'GOOGL', 'AMZN', 'AAPL', 'MSFT', 'CSCO', 'INTC',
    'CVX', 'XOM', 'SLB', 'COP',
    'JPM', 'WFC', 'USB', 'AXP',
    'WMT', 'TGT', 'HD', 'COST'
]

# -----------------------------
# Subset S&P 500 price data
# -----------------------------
top_sp1 = sp500_px.loc[sp500_px.index >= '2005-01-01', syms]

# -----------------------------
# Fit PCA
# -----------------------------
sp_pca1 = PCA()
sp_pca1.fit(top_sp1)

# -----------------------------
# Explained variance
# -----------------------------
explained_variance = pd.DataFrame(sp_pca1.explained_variance_)

# -----------------------------
# Plot the first 10 components
# -----------------------------
ax = explained_variance.head(10).plot.bar(
    legend=False,
    figsize=(4, 4)
)

ax.set_xlabel('Component')
ax.set_ylabel('Explained Variance')
plt.title('Explained Variance of Principal Components')
plt.tight_layout()
plt.show()


loadings = pd.DataFrame(sp_pca1.components_[0:2, :], columns=top_sp1.columns)
print(loadings.transpose())
