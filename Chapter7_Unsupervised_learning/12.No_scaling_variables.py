# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter

# Load dataset
loan_data = pd.read_csv('loan_data.csv')

# Filter for defaults
defaults = loan_data.loc[loan_data['outcome'] == 'default', ]

# Columns to use for clustering
columns = ['loan_amnt', 'annual_inc', 'revol_bal', 'open_acc', 'dti', 'revol_util']
df = defaults[columns]

# Apply K-Means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=1).fit(df)

# Count the number of points in each cluster
counts = Counter(kmeans.labels_)

# Get cluster centers
centers = pd.DataFrame(kmeans.cluster_centers_, columns=columns)

# Add cluster sizes
centers['size'] = [counts[i] for i in range(4)]

# Display the cluster centers with sizes
print(centers)
