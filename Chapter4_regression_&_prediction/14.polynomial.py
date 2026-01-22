import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load data
# -----------------------------
house = pd.read_csv('house_sales.csv', sep='\t')

# Subset ZipCode 98105
house_98105 = house[house['ZipCode'] == 98105]

# -----------------------------
# Polynomial regression model
# -----------------------------
model_poly = smf.ols(
    formula='AdjSalePrice ~ SqFtTotLiving + I(SqFtTotLiving**2)',
    data=house_98105
)
result_poly = model_poly.fit()
print(result_poly.summary())

# -----------------------------
# Create prediction grid
# -----------------------------
x_grid = np.linspace(
    house_98105['SqFtTotLiving'].min(),
    house_98105['SqFtTotLiving'].max(),
    200
)

poly_pred = result_poly.predict(
    pd.DataFrame({'SqFtTotLiving': x_grid})
)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 5))

# Scatter plot of data
plt.scatter(
    house_98105['SqFtTotLiving'],
    house_98105['AdjSalePrice'],
    alpha=0.2
)

# Polynomial regression (solid line)
plt.plot(
    x_grid,
    poly_pred,
    color='black',
    linewidth=2,
    label='Polynomial fit'
)

# Smooth LOWESS curve (dashed line)
sns.regplot(
    x='SqFtTotLiving',
    y='AdjSalePrice',
    data=house_98105,
    lowess=True,
    scatter=False,
    line_kws={'linestyle': '--', 'color': 'red'},
    color='red',
    label='Smooth (LOWESS)'
)

plt.xlabel('SqFtTotLiving')
plt.ylabel('AdjSalePrice')
plt.title('Polynomial vs Smooth Fit for SqFtTotLiving')
plt.legend()
plt.show()
