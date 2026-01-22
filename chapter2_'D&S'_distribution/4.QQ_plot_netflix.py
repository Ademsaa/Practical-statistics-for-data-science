import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the data
sp500_px = pd.read_csv('sp500_data.csv', index_col=0, parse_dates=True)

# Extract Netflix prices
nflx_prices = sp500_px['NFLX']

# Remove zeros or negative values if any
nflx_prices = nflx_prices[nflx_prices > 0]

# Calculate log returns
nflx_returns = np.diff(np.log(nflx_prices.values))

# Create QQ-Plot
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(nflx_returns, dist="norm", plot=ax)
ax.set_title("QQ-Plot of Netflix (NFLX) Log Returns")
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
plt.show()
