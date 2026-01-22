import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load data
house = pd.read_csv('house_sales.csv', sep='\t')

# Define predictors and outcome for initial model
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# Fit initial linear regression
house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])

# Check value counts of ZipCode
print(pd.DataFrame(house['ZipCode'].value_counts()).transpose())

# Create dataframe with residuals per ZipCode
zip_groups = pd.DataFrame([
    *pd.DataFrame({
        'ZipCode': house['ZipCode'],
        'residual': house[outcome] - house_lm.predict(house[predictors])})
    
    .groupby('ZipCode')
    .apply(lambda x: {
        'ZipCode': x.iloc[0, 0],
        'count': len(x),
        'median_residual': x['residual'].median()})]).sort_values('median_residual')

# Compute cumulative count
zip_groups['cum_count'] = np.cumsum(zip_groups['count'])

# Create 5 quantile-based groups
zip_groups['ZipGroup'] = pd.qcut(zip_groups['cum_count'], 5, labels=False, retbins=False)

# Join ZipGroup back to the house dataframe
to_join = zip_groups[['ZipCode', 'ZipGroup']].set_index('ZipCode')
house = house.join(to_join, on='ZipCode')

# Convert ZipGroup to categorical
house['ZipGroup'] = house['ZipGroup'].astype('category')

# Inspect the result
print(house[['ZipCode', 'ZipGroup']])

#=====================================================================
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
'BldgGrade', 'PropertyType', 'ZipGroup']
outcome = 'AdjSalePrice'
X = pd.get_dummies(house[predictors], drop_first=True)
confounding_lm = LinearRegression()
confounding_lm.fit(X, house[outcome])



print(f'Intercept: {confounding_lm.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(X.columns, confounding_lm.coef_):
    print(f' {name}: {coef}')
