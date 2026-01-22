# --------------------------------------------------
# 1. Import libraries
# --------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# --------------------------------------------------
# 2. Load S&P 500 data
# --------------------------------------------------
sp500_px = pd.read_csv("sp500_data.csv", index_col=0, parse_dates=True)

# --------------------------------------------------
# 3. Select two stocks and filter dates
# --------------------------------------------------
df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]

# Optional: standardize returns if needed
# df = (df - df.mean()) / df.std()

# --------------------------------------------------
# 4. Fit Gaussian Mixture Model
# --------------------------------------------------
mclust = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
mclust.fit(df.values)

# Compute BIC
bic_value = mclust.bic(df.values)
print("BIC:", bic_value)

# --------------------------------------------------
# 5. Predict cluster labels
# --------------------------------------------------
labels = mclust.predict(df.values)

# --------------------------------------------------
# 6. Plot clusters
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
colors = [f'C{c}' for c in labels]
df.plot.scatter(x='XOM', y='CVX', c=colors, alpha=0.5, ax=ax)

ax.set_xlabel('XOM Returns')
ax.set_ylabel('CVX Returns')
ax.set_title('Gaussian Mixture Clustering of XOM & CVX')
ax.set_xlim(-3, 3)  # adjust based on your scaled/returns data
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.show()

#
print('Mean')
print(mclust.means_)
print('Covariances')
print(mclust.covariances_)