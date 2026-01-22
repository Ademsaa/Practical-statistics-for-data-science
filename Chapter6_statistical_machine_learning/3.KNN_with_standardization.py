from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd

# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------
loan = pd.read_csv("loan_data.csv")

# --------------------------------------------------
# 2. Define predictors (features) and outcome (label)
# --------------------------------------------------
predictors = ['payment_inc_ratio', 'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'

# --------------------------------------------------
# 3. Define the new loan to analyze (first row)
# --------------------------------------------------
newloan = loan.loc[0:0, predictors]

# --------------------------------------------------
# 4. Prepare training data (exclude the new loan)
# --------------------------------------------------
X = loan.loc[1:, predictors]   # predictor variables
y = loan.loc[1:, outcome]      # loan outcomes

# --------------------------------------------------
# 5. Standardize the predictor variables
#    (KNN is distance-based, so scaling is essential)
# --------------------------------------------------
scaler = preprocessing.StandardScaler()

# Fit the scaler on training data
scaler.fit(X.astype(float))

# Transform both training data and new loan
X_std = scaler.transform(X.astype(float))
newloan_std = scaler.transform(newloan.astype(float))

# --------------------------------------------------
# 6. Create the KNN model with K = 5
# --------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model using the standardized data
knn.fit(X_std, y)

# --------------------------------------------------
# 7. Find the 5 nearest neighbors of the new loan
# --------------------------------------------------
distances, indices = knn.kneighbors(newloan_std)

# --------------------------------------------------
# 8. Display the distances and the nearest neighbors
# --------------------------------------------------
print("Distances to the 5 nearest neighbors:")
print(distances[0])

print("\nPredictor values of the 5 nearest neighbors:")
print(X.iloc[indices[0], :])
