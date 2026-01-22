# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load your dataset
loan_data = pd.read_csv('loan_data.csv')

# Ensure the outcome is categorical
loan_data['outcome'] = pd.Categorical(loan_data['outcome'])

# Define predictors and outcome
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

# Create dummy variables for categorical predictors
X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = loan_data[outcome]

# Initialize and fit logistic regression model
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)

# Optional: view model coefficients
coefficients = pd.DataFrame({'Predictor': X.columns, 'Coefficient': logit_reg.coef_[0]})
print("Model Coefficients:\n", coefficients)

# Get predicted probabilities for each class
pred = pd.DataFrame(
    logit_reg.predict_proba(X),  # actual probabilities
    columns=loan_data[outcome].cat.categories  # class names
)

# View summary statistics of the predicted probabilities
print("Predicted Probabilities Summary:\n", pred.describe())

# Optional: view first few rows
print("First 5 Predicted Probabilities:\n", pred.head())
