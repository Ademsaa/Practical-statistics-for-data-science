from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# Load the dataset
loan = pd.read_csv("loan_data.csv")


predictors = ['dti', 'revol_bal', 'revol_util', 'open_acc',
'delinq_2yrs_zero', 'pub_rec_zero']
outcome = 'outcome'

X = loan[predictors]
y = loan[outcome]

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y)

loan['borrower_score'] = knn.predict_proba(X)[:, 1]
print(loan['borrower_score'].describe())


