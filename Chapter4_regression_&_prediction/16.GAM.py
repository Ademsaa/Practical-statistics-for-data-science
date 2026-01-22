import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, l
from statsmodels.nonparametric.smoothers_lowess import lowess

# -----------------------------
# 1. Load data
# -----------------------------
house = pd.read_csv('house_sales.csv', sep='\t')

# Subset to ZipCode 98105
house_98105 = house[house['ZipCode'] == 98105]

# -----------------------------
# 2. Define predictors and outcome
# -----------------------------
predictors = [
    'SqFtTotLiving',
    'SqFtLot',
    'Bathrooms',
    'Bedrooms',
    'BldgGrade'
]

outcome = 'AdjSalePrice'

X = house_98105[predictors].values
y = house_98105[outcome].values

# -----------------------------
# 3. Fit GAM model
# -----------------------------
gam = LinearGAM(
    s(0, n_splines=12) +  # nonlinear effect for SqFtTotLiving
    l(1) + l(2) + l(3) + l(4)
)

gam.gridsearch(X, y)

# -----------------------------
# 4. Prediction grid
# -----------------------------
x_grid = np.linspace(
    house_98105['SqFtTotLiving'].min(),
    house_98105['SqFtTotLiving'].max(),
    300
)

X_pred = np.zeros((300, X.shape[1]))
X_pred[:, 0] = x_grid

# hold other variables at their means
for i in range(1, X.shape[1]):
    X_pred[:, i] = X[:, i].mean()

y_gam = gam.predict(X_pred)

# -----------------------------
# 5. LOWESS smooth (dashed line)
# -----------------------------
lowess_fit = lowess(
    y,
    house_98105['SqFtTotLiving'],
    frac=0.3
)

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(6, 5))

# Scatter points
plt.scatter(
    house_98105['SqFtTotLiving'],
    y,
    alpha=0.2
)

# GAM fit (solid line)
plt.plot(
    x_grid,
    y_gam,
    color='black',
    linewidth=2,
    label='GAM fit'
)

# LOWESS smooth (dashed line)
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
plt.title('GAM vs Smooth Fit for SqFtTotLiving')
plt.legend()
plt.show()
