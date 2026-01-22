import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 1. Load the dataset
# -------------------------
loan3000 = pd.read_csv("loan3000.csv")

predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

# -------------------------------
# 2. Fit Random Forest (500 trees)
# -------------------------------
rf_500 = RandomForestClassifier(
    n_estimators=500,
    random_state=1,
    oob_score=True
)
rf_500.fit(X, y)
print("OOB score (500 trees):", rf_500.oob_score_)

# -------------------------------
# 3. Scatter plot of predictions
# -------------------------------
predictions = X.copy()
predictions['prediction'] = rf_500.predict(X)

fig1, ax1 = plt.subplots(figsize=(4, 4))

# Plot predicted 'paid off'
predictions.loc[predictions.prediction == 'paid off'].plot(
    x='borrower_score',
    y='payment_inc_ratio',
    style='.',
    markerfacecolor='none',
    markeredgecolor='C1',
    ax=ax1
)

# Plot predicted 'default'
predictions.loc[predictions.prediction == 'default'].plot(
    x='borrower_score',
    y='payment_inc_ratio',
    style='o',
    markerfacecolor='none',
    markeredgecolor='C0',
    ax=ax1
)

ax1.legend(['paid off', 'default'])
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 25)
ax1.set_xlabel('borrower_score')
ax1.set_ylabel('payment_inc_ratio')
ax1.set_title('Random Forest Predictions')

plt.show()

# -----------------------------------------
# 4. OOB error vs number of trees (book style)
# -----------------------------------------
n_estimators = list(range(20, 510, 5))
oob_error = []

for n in n_estimators:
    rf = RandomForestClassifier(
        n_estimators=n,
        criterion='entropy',
        max_depth=5,
        random_state=1,
        oob_score=True
    )
    rf.fit(X, y)
    # OOB error = 1 - OOB score
    oob_error.append(1 - rf.oob_score_)

# Plot OOB error
plt.figure(figsize=(6, 4))
plt.plot(n_estimators, oob_error, color='black', linewidth=1)
plt.xlabel('Number of Trees')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error vs Number of Trees')
plt.grid(True)
plt.show()
