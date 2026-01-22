# -----------------------------
# Complete code: K-Means with categorical variables
# -----------------------------

# Import necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# -----------------------------
# Filter for defaulted loans
# -----------------------------
# Assuming 'loan_data' is your DataFrame
loan_data = pd.read_csv('loan_data.csv')
defaults = loan_data.loc[loan_data['outcome'] == 'default', ]

# -----------------------------
# Select columns (numeric + categorical)
# -----------------------------
columns = ['dti', 'payment_inc_ratio', 'home_', 'pub_rec_zero']
df = defaults[columns]

# -----------------------------
# Convert categorical variables to numeric using one-hot encoding
# -----------------------------
df = pd.get_dummies(df)  # home_ will be converted to binary columns

# -----------------------------
# Scale the data
# -----------------------------
scaler = preprocessing.StandardScaler()
df0 = scaler.fit_transform(df * 1.0)  # ensure float type

# -----------------------------
# Apply K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=4, random_state=1).fit(df0)

# -----------------------------
# Get cluster centers in original units
# -----------------------------
centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                       columns=df.columns)

# -----------------------------
# Display cluster centers
# -----------------------------
print("Cluster Centers:")
print(centers)
