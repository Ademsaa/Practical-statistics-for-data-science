import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
loan_data = pd.read_csv('loan_data.csv')

# Ensure outcome is categorical
loan_data['outcome'] = loan_data['outcome'].astype('category')

# Define predictors and outcome
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

# Create dummy variables
X = pd.get_dummies(loan_data[predictors], drop_first=True).astype(float)
y = loan_data[outcome]

# Fit logistic regression
# penalty=None replaces C=1e42 for unregularized regression
# 'lbfgs' is the default and robust solver for penalty=None
logit_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
logit_reg.fit(X, y)

# Predictions
pred = logit_reg.predict(X)

# --- Built-in Confusion Matrix ---
# This approach is less prone to manual indexing errors
labels = ['default', 'paid off']
cm = confusion_matrix(y, pred, labels=labels)

conf_mat = pd.DataFrame(
    cm, 
    index=[f'Actual {l}' for l in labels], 
    columns=[f'Predicted {l}' for l in labels]
)

print("Confusion Matrix:")
print(conf_mat)
print("\nClassification Report:")
print(classification_report(y, pred))