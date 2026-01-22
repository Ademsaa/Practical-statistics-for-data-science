import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Load data
loan_data = pd.read_csv('loan_data.csv')

# Prepare outcome (default = 1, paid off = 0)
y = (loan_data['outcome'] == 'default').astype(int)

# Predictors
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 'borrower_score']
X = pd.get_dummies(loan_data[predictors], drop_first=True).astype(float)

# Fit logistic regression
model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=500)
model.fit(X, y)

# Predicted probability of default
y_prob = model.predict_proba(X)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y, y_prob)

# Plot ROC curve (Specificity vs Recall)
plt.figure(figsize=(5, 5))
plt.plot(1 - fpr, tpr)
plt.plot([1, 0], [0, 1], linestyle='--')
plt.xlabel('Specificity')
plt.ylabel('Recall')
plt.xlim(1, 0)
plt.ylim(0, 1)
plt.show()

#print AUC_score
print("AUC:", roc_auc_score(y, y_prob))