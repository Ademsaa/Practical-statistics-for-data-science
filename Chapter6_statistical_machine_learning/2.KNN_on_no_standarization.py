from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load the dataset
loan = pd.read_csv("loan_data.csv")


# Define predictors and outcome
predictors = ['payment_inc_ratio', 'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'

# New loan to predict
newloan = loan.loc[0:0, predictors]

# Training data
X = loan.loc[1:, predictors]
y = loan.loc[1:, outcome]

# KNN model with K = 5
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model
knn.fit(X, y)

# Find the 5 nearest neighbors of the new loan
distances, indices = knn.kneighbors(newloan)

# Display the predictor values of the nearest neighbors
print("Distances:")
print(distances[0])

print("\nNearest neighbors:")
print(X.iloc[indices[0], :])







