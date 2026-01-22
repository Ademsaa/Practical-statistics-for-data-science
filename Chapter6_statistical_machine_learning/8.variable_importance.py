# random_forest_loan.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import defaultdict
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
# 3. Prepare the data
# -----------------------------
X = pd.get_dummies(loan_data[predictors], drop_first=True)
y = loan_data[outcome]

# -----------------------------
# 4. Build and fit the Random Forest model
# -----------------------------
rf_all = RandomForestClassifier(n_estimators=500, random_state=1)
rf_all.fit(X, y)

# Print feature importances from Gini decrease
importances = pd.Series(rf_all.feature_importances_, index=X.columns)
print("Feature importances (Gini decrease):\n", importances.sort_values(ascending=False))

# -----------------------------
# 5. Evaluate feature importance via permutation accuracy decrease
# -----------------------------
rf = RandomForestClassifier(n_estimators=500, random_state=1)
scores = defaultdict(list)

# Cross-validate the scores on a number of different random splits of the data
for _ in range(3):
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.3)
    rf.fit(train_X, train_y)
    acc = metrics.accuracy_score(valid_y, rf.predict(valid_X))
    
    # Permutation: shuffle each feature and measure accuracy drop
    for column in X.columns:
        X_t = valid_X.copy()
        X_t[column] = np.random.permutation(X_t[column].values)
        shuff_acc = metrics.accuracy_score(valid_y, rf.predict(X_t))
        scores[column].append((acc - shuff_acc) / acc)

# -----------------------------
# 6. Create DataFrame for plotting
# -----------------------------
df = pd.DataFrame({
    'feature': X.columns,
    'Accuracy decrease': [np.mean(scores[column]) for column in X.columns],
    'Gini decrease': rf_all.feature_importances_,
})
df = df.sort_values('Accuracy decrease')

# -----------------------------
# 7. Plot feature importances
# -----------------------------
fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

# Accuracy decrease plot
ax = df.plot(kind='barh', x='feature', y='Accuracy decrease', legend=False, ax=axes[0])
ax.set_ylabel('')
ax.set_title('Feature Importance (Accuracy Decrease)')

# Gini decrease plot
ax = df.plot(kind='barh', x='feature', y='Gini decrease', legend=False, ax=axes[1])
ax.set_ylabel('')
ax.set_title('Feature Importance (Gini Decrease)')
ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()
