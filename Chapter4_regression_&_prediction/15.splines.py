import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from patsy import bs

# -----------------------------
# 1. Load data
# -----------------------------
house = pd.read_csv('house_sales.csv', sep='\t')

# Focus on one zip code (as in the book)
house_98105 = house[house['ZipCode'] == 98105]

# -----------------------------
# 2. Fit spline regression
# -----------------------------
formula = (
    'AdjSalePrice ~ bs(SqFtTotLiving, df=6, degree=3) + '
    'SqFtLot + Bathrooms + Bedrooms + BldgGrade'
)

model_spline = smf.ols(formula=formula, data=house_98105)
result_spline = model_spline.fit()
print(result_spline.summary())

# -----------------------------
# 3. Create prediction grid
# -----------------------------
x_grid = np.linspace(
    house_98105['SqFtTotLiving'].min(),
    house_98105['SqFtTotLiving'].max(),
    300
)

pred_df = pd.DataFrame({
    'SqFtTotLiving': x_grid,
    'SqFtLot': house_98105['SqFtLot'].mean(),
    'Bathrooms': house_98105['Bathrooms'].mean(),
    'Bedrooms': house_98105['Bedrooms'].mean(),
    'BldgGrade': house_98105['BldgGrade'].mean()
})

# Spline predictions (solid line)
y_spline = result_spline.predict(pred_df)

# -----------------------------
# 4. LOWESS smooth (dashed line)
# -----------------------------
lowess_fit = lowess(
    house_98105['AdjSalePrice'],
    house_98105['SqFtTotLiving'],
    frac=0.3
)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(6, 5))

# Scatter points
plt.scatter(
    house_98105['SqFtTotLiving'],
    house_98105['AdjSalePrice'],
    alpha=0.2
)

# Spline regression (solid)
plt.plot(
    x_grid,
    y_spline,
    color='black',
    linewidth=2,
    label='Spline regression'
)

# LOWESS smooth (dashed)
plt.plot(
    lowess_fit[:, 0],
    lowess_fit[:, 1],
    linestyle='--',
    color='black',
    linewidth=2,
    label='Smooth (LOWESS)'
)

plt.xlabel('SqFtTotLiving')
plt.ylabel('AdjSalePrice')
plt.title('Spline vs Smooth Fit for SqFtTotLiving')
plt.legend()
plt.show()
