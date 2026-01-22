# xgboost_ntree_limit_plot.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = pd.Series([1 if o == 'default' else 0 for o in loan_data[outcome]])

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=10000, random_state=0)

# -----------------------------
# 4. Train default XGBoost model
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
# 5. Train penalized XGBoost model
# -----------------------------
xgb_penalty = XGBClassifier(
    objective='binary:logistic',
    n_estimators=250,
    max_depth=6,
    reg_lambda=1000,   # L2 regularization
    learning_rate=0.1,
    subsample=0.63,
    random_state=0,
    verbosity=1
)
xgb_penalty.fit(train_X, train_y)

# -----------------------------
# 6. Evaluate error at each iteration (iteration_range replaces ntree_limit)
# -----------------------------
results = []

for i in range(1, 250):
    # Train predictions
    train_default = xgb_default.predict_proba(train_X, iteration_range=(0, i))[:, 1]
    train_penalty = xgb_penalty.predict_proba(train_X, iteration_range=(0, i))[:, 1]
    
    # Validation predictions
    pred_default = xgb_default.predict_proba(valid_X, iteration_range=(0, i))[:, 1]
    pred_penalty = xgb_penalty.predict_proba(valid_X, iteration_range=(0, i))[:, 1]
    
    # Record errors using 0.5 threshold
    results.append({
        'iterations': i,
        'default train': np.mean(abs(train_y - (train_default > 0.5).astype(int))),
        'penalty train': np.mean(abs(train_y - (train_penalty > 0.5).astype(int))),
        'default test': np.mean(abs(valid_y - (pred_default > 0.5).astype(int))),
        'penalty test': np.mean(abs(valid_y - (pred_penalty > 0.5).astype(int))),
    })

results = pd.DataFrame(results)

# -----------------------------
# 7. Plot training vs test errors
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
results.plot(x='iterations', y='default test', ax=ax, color='red', linestyle='--')
results.plot(x='iterations', y='penalty test', ax=ax, color='blue', linestyle='--')
results.plot(x='iterations', y='default train', ax=ax, color='red', linestyle='-')
results.plot(x='iterations', y='penalty train', ax=ax, color='blue', linestyle='-')

ax.set_xlabel('Number of Trees')
ax.set_ylabel('Error Rate')
ax.set_title('XGBoost Training vs Test Error Across Iterations')
ax.legend(['Default Test', 'Penalty Test', 'Default Train', 'Penalty Train'])
plt.grid(True)
plt.show()