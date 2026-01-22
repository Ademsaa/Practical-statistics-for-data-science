import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
house = pd.read_csv('house_sales.csv', sep='\t')

# Create a weight column based on year
house['Year'] = house['DocumentDate'].apply(lambda x: int(x.split('-')[0]))
house['Weight'] = house['Year'] - 2005  # Example: weight = years since 2005

# Define predictors and outcome
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# Fit weighted linear regression
house_wt = LinearRegression()
house_wt.fit(house[predictors], house[outcome], sample_weight=house['Weight'])

# Print coefficients
print(f'Intercept: {house_wt.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(predictors, house_wt.coef_):
    print(f' {name}: {coef:.3f}')
