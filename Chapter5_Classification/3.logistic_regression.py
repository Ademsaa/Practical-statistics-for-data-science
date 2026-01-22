# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load your dataset (replace 'loan_data.csv' with your actual file if needed)
loan_data = pd.read_csv('loan_data.csv')

# Define predictors and outcome
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

# Create dummy variables for categorical predictors
X = pd.get_dummies(loan_data[predictors], prefix='', prefix_sep='', drop_first=True)
y = loan_data[outcome]

# Initialize and fit logistic regression model
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)

# Optional: view model coefficients
coefficients = pd.DataFrame({'Predictor': X.columns, 'Coefficient': logit_reg.coef_[0]})
print(coefficients)
