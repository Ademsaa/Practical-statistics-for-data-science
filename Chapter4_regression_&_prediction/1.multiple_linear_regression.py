import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
# Read the tab-delimited CSV
house = pd.read_csv('house_sales.csv', sep='\t')

# Select subset of columns
subset = ['AdjSalePrice', 'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']

# Display first 5 rows of the subset
print(house[subset].head())

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])


print(f'Intercept: {house_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(predictors, house_lm.coef_):
    print(f' {name}: {coef}')

#============================================================
# Make predictions
fitted = house_lm.predict(house[predictors])
# Calculate RMSE and R-squared
RMSE = np.sqrt(mean_squared_error(house[outcome], fitted))
r2 = r2_score(house[outcome], fitted)
# Print results
print(f'RMSE: {RMSE:.0f}')
print(f'r2: {r2:.4f}')

#============================================================
# Add constant for intercept
X = house[predictors].assign(const=1)
y = house[outcome]

# Fit OLS model
model = sm.OLS(y, X)
results = model.fit()

# Show summary
print(results.summary())

#===========================================
predictors = [
    'SqFtTotLiving',
    'SqFtLot',
    'Bathrooms',
    'Bedrooms',
    'BldgGrade',
    'PropertyType',
    'NbrLivingUnits',
    'SqFtFinBasement',
    'YrBuilt',
    'YrRenovated',
    'NewConstruction'
]
outcome = 'AdjSalePrice'

# Create design matrix with dummy variables
X = pd.get_dummies(house[predictors], drop_first=True)

# Ensure NewConstruction is coded as 0/1
X['NewConstruction'] = X['NewConstruction'].apply(lambda nc: 1 if nc else 0)

# Add intercept
X = sm.add_constant(X)

# Fit OLS regression
house_full = sm.OLS(house[outcome], X)
results = house_full.fit()

# Show regression summary
results.summary()