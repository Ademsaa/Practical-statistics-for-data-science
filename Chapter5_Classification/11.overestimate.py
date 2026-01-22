import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load the dataset
full_train_set = pd.read_csv('loan_data.csv')

# Calculate percentage of loans in default (actual)
percent_default = 100 * np.mean(full_train_set['outcome'] == 'default')
print('Percentage of loans in default:', percent_default)

# Define predictors and outcome
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 
              'dti', 'revol_bal', 'revol_util']
outcome = 'outcome'

# Convert categorical predictors to dummy variables
X = pd.get_dummies(full_train_set[predictors], drop_first=True).astype(float)
y = full_train_set[outcome]

# Compute sample weights for the rare class (defaults)
default_wt = 1 / np.mean(full_train_set[outcome] == 'default')
wt = [default_wt if val == 'default' else 1 for val in full_train_set[outcome]]

# Fit logistic regression with sample weights
full_model = LogisticRegression(penalty="l2", C=1e42, solver='liblinear', max_iter=500)
full_model.fit(X, y, sample_weight=wt)

# Percentage of loans predicted to default
percent_pred_default = 100 * np.mean(full_model.predict(X) == 'default')
print('Percentage of loans predicted to default (weighting):', percent_pred_default)
