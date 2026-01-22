# xgboost_loan3000.py

import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the dataset
# -----------------------------
loan3000 = pd.read_csv('loan3000.csv')

# -----------------------------
# 2. Define predictors and outcome
# -----------------------------
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

# -----------------------------
# 3. Encode target variable as numeric
# -----------------------------
y_numeric = y.map({'paid off': 0, 'default': 1})

# -----------------------------
# 4. Build the XGBoost model
# -----------------------------
xgb = XGBClassifier(
    objective='binary:logistic',
    subsample=0.63,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=0,
    verbosity=1
)

# -----------------------------
# 5. Fit the model
# -----------------------------
xgb.fit(X, y_numeric)

# -----------------------------
# 6. Make predictions and create DataFrame for plotting
# -----------------------------
# Predict class labels
pred_labels = xgb.predict(X)

# Map numeric predictions back to strings for plotting
pred_labels_str = pd.Series(pred_labels).map({0: 'paid off', 1: 'default'})

# Combine predictions with predictors into one DataFrame
xgb_df = X.copy()
xgb_df['prediction'] = pred_labels_str

# -----------------------------
# 7. Plot predictions
# -----------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Plot borrowers predicted 'paid off'
xgb_df.loc[xgb_df.prediction == 'paid off'].plot(
    x='borrower_score', y='payment_inc_ratio', style='.',
    markerfacecolor='none', markeredgecolor='C1', ax=ax
)

# Plot borrowers predicted 'default'
xgb_df.loc[xgb_df.prediction == 'default'].plot(
    x='borrower_score', y='payment_inc_ratio', style='o',
    markerfacecolor='none', markeredgecolor='C0', ax=ax
)

# Customize plot
ax.legend(['paid off', 'default'])
ax.set_xlim(0, 1)
ax.set_ylim(0, 25)
ax.set_xlabel('borrower_score')
ax.set_ylabel('payment_inc_ratio')
ax.set_title('XGBoost Predictions')

plt.show()
