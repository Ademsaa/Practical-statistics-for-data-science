import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
loan3000 = pd.read_csv("loan3000.csv")

# Convert outcome to categorical
loan3000['outcome'] = loan3000['outcome'].astype('category')

# Define predictors and outcome
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

# Fit LDA model
loan_lda = LinearDiscriminantAnalysis()
loan_lda.fit(X, y)

# Fisher LDA coefficients (scalings)
lda_coefficients = pd.DataFrame(
    loan_lda.scalings_,
    index=X.columns,
    columns=['LDA1']
)
print("LDA coefficients:")
print(lda_coefficients)

# Predicted probabilities for each class
pred = pd.DataFrame(
    loan_lda.predict_proba(X),
    columns=loan_lda.classes_
)

print("\nPredicted probabilities:")
print(pred.head())

# Add predicted probability of default to DataFrame
lda_df = pd.concat([loan3000, pred['default']], axis=1)

# Compute LDA decision boundary
center = loan_lda.means_.mean(axis=0)
slope = -loan_lda.scalings_[0, 0] / loan_lda.scalings_[1, 0]
intercept = center[1] - center[0] * slope

# payment_inc_ratio for borrower_score of 0 and 20
x_0 = (0 - intercept) / slope
x_20 = (20 - intercept) / slope

# Plot
fig, ax = plt.subplots(figsize=(7, 7))
g = sns.scatterplot(x='borrower_score', y='payment_inc_ratio',
                    hue='default', data=lda_df,
                    palette=sns.diverging_palette(240, 10, n=9, as_cmap=True),
                    ax=ax, legend=False)

ax.set_ylim(0, 20)
ax.set_xlim(0.15, 0.8)
ax.plot((x_0, x_20), (0, 20), linewidth=3)
ax.plot(*loan_lda.means_.transpose())

# Add colorbar for probability of default
lda_df['prob_default'] = pred['default']
scatter = ax.scatter(
    lda_df['borrower_score'],
    lda_df['payment_inc_ratio'],
    c=lda_df['prob_default'],               # color points by probability of default
    cmap=sns.diverging_palette(240, 10, n=9, as_cmap=True),  # same palette as before
    alpha=0.7
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Probability of Default')

plt.show()