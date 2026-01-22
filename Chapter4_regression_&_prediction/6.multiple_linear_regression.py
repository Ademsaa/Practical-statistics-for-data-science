import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
house = pd.read_csv('house_sales.csv', sep='\t')
outcome = 'AdjSalePrice'

# Define predictors
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade', 'PropertyType']

# Convert categorical variable into dummy variables, dropping the first level
X = pd.get_dummies(house[predictors], drop_first=True)

# Fit linear regression model
house_lm_factor = LinearRegression()
house_lm_factor.fit(X, house[outcome])

# Print results
print(f'Intercept: {house_lm_factor.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(X.columns, house_lm_factor.coef_):
    print(f' {name}: {coef:.3f}')
