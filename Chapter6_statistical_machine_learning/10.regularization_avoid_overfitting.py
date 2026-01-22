# xgboost_full_model.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Load the dataset
# -----------------------------
loan_data = pd.read_csv('loan_data.csv')

# -----------------------------
# 2. Define predictors and outcome
# -----------------------------
predictors = [
    'loan_amnt', 'term', 'annual_inc', 'dti', 'payment_inc_ratio',
    'revol_bal', 'revol_util', 'purpose', 'delinq_2yrs_zero',
    'pub_rec_zero', 'open_acc', 'grade', 'emp_length', 'purpose_',
    'home_', 'emp_len_', 'borrower_score'
]
outcome = 'outcome'

# -----------------------------
# 3. Prepare data
# -----------------------------
# Convert categorical variables using one-hot encoding
X = pd.get_dummies(loan_data[predictors], drop_first=True)

# Encode outcome as 0/1
y = pd.Series([1 if o == 'default' else 0 for o in loan_data[outcome]])

# Split into training and validation sets
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=10000, random_state=0)

# -----------------------------
# 4. Build and fit the XGBoost model
# -----------------------------
xgb_default = XGBClassifier(
    objective='binary:logistic',
    n_estimators=250,
    max_depth=6,
    reg_lambda=0,
    learning_rate=0.3,
    subsample=1,
    random_state=0,
    verbosity=1
)

xgb_default.fit(train_X, train_y)

# -----------------------------
# 5. Make predictions on validation set
# -----------------------------
# Predict probabilities for the positive class (default)
pred_default = xgb_default.predict_proba(valid_X)[:, 1]

# -----------------------------
# 6. Calculate error rate
# -----------------------------
# Consider a prediction wrong if probability >0.5 is on the wrong side
error_default = abs(valid_y - (pred_default > 0.5).astype(int))
print('Default error rate: ', np.mean(error_default))
