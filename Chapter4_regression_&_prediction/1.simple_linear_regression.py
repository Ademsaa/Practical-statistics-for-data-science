import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
lung = pd.read_csv('LungDisease.csv')

# Define predictors and outcome
predictors = ['Exposure']
outcome = 'PEFR'

# Fit linear regression model
model = LinearRegression()
model.fit(lung[predictors], lung[outcome])

# Print model parameters
print(f'Intercept: {model.intercept_:.3f}')
print(f'Coefficient Exposure: {model.coef_[0]:.3f}')

# ---- Added code ----
# Get fitted (predicted) values
fitted = model.predict(lung[predictors])

# Calculate residuals
residuals = lung[outcome] - fitted

# Optional: inspect results
print('\nFirst 5 fitted values:')
print(fitted[:5])

print('\nFirst 5 residuals:')
print(residuals.head())

