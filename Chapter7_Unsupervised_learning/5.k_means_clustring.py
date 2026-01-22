# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
# Read S&P 500 stock data from CSV file
sp500_px = pd.read_csv("sp500_data.csv", index_col=0, parse_dates=True)

# --------------------------------------------------
# 2. Select data for clustering
# --------------------------------------------------
# Keep data from 2011 onward and select XOM and CVX returns
df = sp500_px.loc[sp500_px.index >= '2011-01-01', ['XOM', 'CVX']]

# --------------------------------------------------
# 3. Apply K-means clustering
# --------------------------------------------------
# Initialize KMeans with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)

# Fit the model to the data
kmeans.fit(df)

# Assign each observation to a cluster
df['cluster'] = kmeans.labels_
print(df.head())

# --------------------------------------------------
# 4. Extract cluster centers
# --------------------------------------------------
# Convert cluster centers to a DataFrame for plotting
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=['XOM', 'CVX']
)
print(centers)

# --------------------------------------------------
# 5. Visualize clusters
# --------------------------------------------------
# Create a scatter plot of the clustered data
fig, ax = plt.subplots(figsize=(4, 4))

sns.scatterplot(
    x='XOM',
    y='CVX',
    hue='cluster',          # Color points by cluster
    style='cluster',        # Marker style by cluster
    data=df,
    ax=ax
)

# Set axis limits for better visualization
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

# Plot cluster centers
centers.plot.scatter(
    x='XOM',
    y='CVX',
    ax=ax,
    s=50,                  # Size of center points
    color='black',         # Centers shown in black
    label='Centroids'
)

# Display the plot
plt.show()
