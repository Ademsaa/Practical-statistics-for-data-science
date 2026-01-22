import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Load the data
loan_data = pd.read_csv("loan_data.csv", sep=',')

# Predictors and outcome
predictors = ['purpose_', 'home_', 'emp_len_']
outcome = 'outcome'

# One-hot encode categorical predictors
X = pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='')
y = loan_data[outcome]

# Initialize and fit the Naive Bayes model
naive_model = MultinomialNB(alpha=0.01, fit_prior=True)
naive_model.fit(X, y)

# Predict for a new loan record (row 146)
new_loan = X.loc[146:146, :]
predicted_class = naive_model.predict(new_loan)[0]
print('Predicted class:', predicted_class)

# Predicted probabilities for each class
probabilities = pd.DataFrame(
    naive_model.predict_proba(new_loan),
    columns=naive_model.classes_   # Use the model's classes instead of .cat.categories
)
print('Predicted probabilities:\n', probabilities)
