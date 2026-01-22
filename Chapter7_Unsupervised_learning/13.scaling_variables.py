# -----------------------------
# Complete Python code for scaled K-Means clustering
# -----------------------------

# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import Counter

# -----------------------------
# Load your data (replace with your file path if using CSV)
# -----------------------------
loan_data = pd.read_csv('loan_data.csv')
# For demonstration, we assume loan_data is already loaded
# It should have columns like 'outcome', 'loan_amnt', 'annual_inc', etc.

# -----------------------------
# Filter the data for defaults
# -----------------------------
defaults = loan_data.loc[loan_data['outcome'] == 'default', ]

# Columns to use for clustering
columns = ['loan_amnt', 'annual_inc', 'revol_bal', 'open_acc', 'dti', 'revol_util']
df = defaults[columns]

# -----------------------------
# Scale the data
# -----------------------------
scaler = preprocessing.StandardScaler()
df0 = scaler.fit_transform(df * 1.0)  # ensure float type

# -----------------------------
# Apply K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=1).fit(df0)

# Count the number of loans in each cluster
counts = Counter(kmeans.labels_)

# -----------------------------
# Get cluster centers and convert back to original units
# -----------------------------
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=columns)

# Add cluster sizes
centers['size'] = [counts[i] for i in range(4)]

# -----------------------------
# Display the results
# -----------------------------
print("Cluster Centers and Sizes:")
print(centers)
