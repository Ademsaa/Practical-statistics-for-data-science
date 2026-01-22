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
# Select two stocks and filter dates
# --------------------------------------------------
df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]


# --------------------------------------------------
# 3. Compute BIC for multiple GMM models
# --------------------------------------------------
results = []
covariance_types = ['full', 'tied', 'diag', 'spherical']

for n_components in range(1, 9):
    for covariance_type in covariance_types:
        mclust = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            warm_start=True,
            random_state=42
        )
        mclust.fit(df)
        results.append({
            'bic': mclust.bic(df),
            'n_components': n_components,
            'covariance_type': covariance_type,
        })

# Convert results to DataFrame
results = pd.DataFrame(results)

# --------------------------------------------------
# 4. Plot BIC values
# --------------------------------------------------
colors = ['C0', 'C1', 'C2', 'C3']
styles = ['-', ':', '-.', '--']

fig, ax = plt.subplots(figsize=(6, 4))

for i, covariance_type in enumerate(covariance_types):
    subset = results.loc[results.covariance_type == covariance_type, :]
    subset.plot(
        x='n_components',
        y='bic',
        ax=ax,
        label=covariance_type,
        kind='line',
        style=styles[i]
    )

ax.set_title('BIC for different GMM models')
ax.set_xlabel('Number of Components')
ax.set_ylabel('BIC')
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 5. Optional: find best model
# --------------------------------------------------
best_model = results.loc[results['bic'].idxmin()]
print("Best model found:")
print(best_model)
