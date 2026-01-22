from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load the dataset
loan200 = pd.read_csv("loan200.csv")
# Display first rows to verify
print(loan200.head())


# Define predictors and outcome
predictors = ['payment_inc_ratio', 'dti']
outcome = 'outcome'

# New loan to predict
newloan = loan200.loc[0:0, predictors]

# Training data
X = loan200.loc[1:, predictors]
y = loan200.loc[1:, outcome]

# KNN model with K = 20
knn = KNeighborsClassifier(n_neighbors=20)

# Fit the model
knn.fit(X, y)

# Predict outcome for the new loan
kpred = knn.predict(newloan)
print(f'the predicted : {kpred}')
