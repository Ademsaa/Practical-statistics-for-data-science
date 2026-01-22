# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
oil_px = pd.read_csv("sp500_data.csv")  # replace with your file path if needed

# Keep only numeric columns (exclude date or text columns)
oil_px_numeric = oil_px.select_dtypes(include='number')

# Initialize PCA with 2 components
pcs = PCA(n_components=2)

# Fit PCA on the numeric data
pcs.fit(oil_px_numeric)

# Create a DataFrame of the component loadings
loadings = pd.DataFrame(pcs.components_, columns=oil_px_numeric.columns, index=['PC1', 'PC2'])

# Display the loadings
print("PCA Component Loadings:")
print(loadings)

# Function to calculate line coordinates based on slope and intercept
def abline(slope, intercept, ax):
    """Calculate coordinates of a line based on slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    return x_vals, intercept + slope * x_vals


# Scatter plot of CVX vs XOM
ax = oil_px_numeric.plot.scatter(x='CVX', y='XOM', alpha=0.3, figsize=(4, 4))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Plot the first principal component direction
# Note: slope = delta_y / delta_x = XOM / CVX now
ax.plot(*abline(loadings.loc['PC1', 'XOM'] / loadings.loc['PC1', 'CVX'], 0, ax), '--', color='C1', label='PC1')

# Plot the second principal component direction
ax.plot(*abline(loadings.loc['PC2', 'XOM'] / loadings.loc['PC2', 'CVX'], 0, ax), '--', color='C2', label='PC2')

# Add legend
ax.legend()
plt.show()

# Assuming PCA has been fitted as pcs and loadings DataFrame exists

# Print the weights for XOM and CVX
print("Weights (loadings) for the first principal component (PC1):")
print(f"XOM: {loadings.loc['PC1', 'XOM']:.3f}, CVX: {loadings.loc['PC1', 'CVX']:.3f}")

print("\nWeights (loadings) for the second principal component (PC2):")
print(f"XOM: {loadings.loc['PC2', 'XOM']:.3f}, CVX: {loadings.loc['PC2', 'CVX']:.3f}")

# Interpretation
print("\nInterpretation:")
print("PC1 is essentially an average of CVX and XOM, reflecting their correlation.")
print("PC2 measures when the stock prices of CVX and XOM diverge from each other.")
