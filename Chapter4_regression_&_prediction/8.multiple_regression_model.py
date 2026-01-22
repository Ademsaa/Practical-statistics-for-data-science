import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
house = pd.read_csv('house_sales.csv', sep='\t')

# Define the modified list of predictors
predictors = ['Bedrooms', 'BldgGrade', 'PropertyType', 'YrBuilt']
outcome = 'AdjSalePrice'

# Convert categorical variables to dummy variables
X = pd.get_dummies(house[predictors], drop_first=True)

# Fit the linear regression model
reduced_lm = LinearRegression()
reduced_lm.fit(X, house[outcome])

# Print model parameters
print(f'Intercept: {reduced_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(X.columns, reduced_lm.coef_):
    print(f' {name}: {coef:.3f}')
