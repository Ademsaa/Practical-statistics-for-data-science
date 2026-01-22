import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

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

# Fit logistic regression (unregularized)
logit_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
logit_reg.fit(X, y)

# Predictions
pred = logit_reg.predict(X)

# Confusion matrix with fixed label order
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
print(classification_report(y, pred, labels=labels))

# ---- Manual metric calculation (for 'default' = positive class) ----
TP = cm[0, 0]
FN = cm[0, 1]
FP = cm[1, 0]
TN = cm[1, 1]

precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)

print("\nManual Metrics (default as positive class):")
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)

# sklearn metric breakdown
precision_recall_fscore_support(y, pred, labels=labels)

