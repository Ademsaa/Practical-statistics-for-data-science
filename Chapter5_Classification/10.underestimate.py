import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset
loan_data = pd.read_csv('loan_data.csv')

# Calculate percentage of loans in default (actual)
percent_default = 100 * np.mean(loan_data['outcome'] == 'default')
print('Percentage of loans in default:', percent_default)

# Define predictors and outcome
predictors = [
    'payment_inc_ratio', 'purpose_', 'home_', 'emp_len_',
    'dti', 'revol_bal', 'revol_util'
]
outcome = 'outcome'

# Create dummy variables for predictors
X = pd.get_dummies(
    loan_data[predictors],
    prefix='',
    prefix_sep='',
    drop_first=True
).astype(float)

y = loan_data[outcome]

# Fit logistic regression
full_model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
full_model.fit(X, y)

# Percentage of loans predicted to default
percent_pred_default = 100 * np.mean(full_model.predict(X) == 'default')
print('Percentage of loans predicted to default:', percent_pred_default)
