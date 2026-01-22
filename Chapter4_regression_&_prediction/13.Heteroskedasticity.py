import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
house = pd.read_csv('house_sales.csv', sep='\t')

# -----------------------------
# 2. Subset data for ZipCode 98105
# -----------------------------
house_98105 = house.loc[house['ZipCode'] == 98105]

# -----------------------------
# 3. Define predictors and outcome
# -----------------------------
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# -----------------------------
# 4. Prepare design matrix (add intercept)
# -----------------------------
X = house_98105[predictors].assign(const=1)
y = house_98105[outcome]

# -----------------------------
# 5. Fit OLS regression model
# -----------------------------
model_98105 = sm.OLS(y, X)
result_98105 = model_98105.fit()

# -----------------------------
# 6. Residuals vs fitted values plot
# -----------------------------
fig, ax = plt.subplots(figsize=(5, 5))

sns.regplot(
    x=result_98105.fittedvalues,
    y=np.abs(result_98105.resid),
    scatter_kws={'alpha': 0.25},
    line_kws={'color': 'C1'},
    lowess=True,
    ax=ax
)

ax.set_xlabel('Predicted values')
ax.set_ylabel('Absolute residuals')
ax.set_title('Residual Diagnostics (ZipCode 98105)')

plt.show()

# -----------------------------
# 7. CCPR plot for SqFtTotLiving
# -----------------------------
sm.graphics.plot_ccpr(result_98105, 'SqFtTotLiving')
plt.show()
