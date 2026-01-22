import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from itertools import product

# Load dataset
loan_data = pd.read_csv('loan_data.csv')

# Define predictors and target
predictors = [
    'loan_amnt', 'term', 'annual_inc', 'dti', 'payment_inc_ratio',
    'revol_bal', 'revol_util', 'purpose', 'delinq_2yrs_zero',
    'pub_rec_zero', 'open_acc', 'grade', 'emp_length', 'purpose_',
    'home_', 'emp_len_', 'borrower_score'
]
outcome = 'outcome'

# Prepare data
X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = pd.Series([1 if o == 'default' else 0 for o in loan_data[outcome]])

# 5-Fold CV Grid Search
np.random.seed(42)
idx = np.random.choice(range(5), size=len(X), replace=True)
error = []

for eta, max_depth in product([0.1, 0.5, 0.9], [3, 6, 9]):
    xgb = XGBClassifier(
        objective='binary:logistic',
        n_estimators=250,
        max_depth=max_depth,
        learning_rate=eta,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    cv_error = []
    for k in range(5):
        fold_idx = idx == k
        train_X = X.loc[~fold_idx]
        train_y = y[~fold_idx]
        valid_X = X.loc[fold_idx]
        valid_y = y[fold_idx]
        
        xgb.fit(train_X, train_y)
        pred = xgb.predict_proba(valid_X)[:, 1]
        cv_error.append(np.mean(abs(valid_y - (pred > 0.5).astype(int))))
    
    error.append({
        'eta': eta,
        'max_depth': max_depth,
        'avg_error': np.mean(cv_error)
    })
    print(error[-1])

errors = pd.DataFrame(error)
print(errors)
