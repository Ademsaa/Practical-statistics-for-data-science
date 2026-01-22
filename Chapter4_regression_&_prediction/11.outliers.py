import pandas as pd
import statsmodels.api as sm

# Load data
house = pd.read_csv('house_sales.csv', sep='\t')

# Subset datae'
#This creates a DataFrame that has:
#All columns from house
#Only rows where ZipCode == 98105
house_98105 = house.loc[house['ZipCode'] == 98105].dropna()

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

X = sm.add_constant(house_98105[predictors])
y = house_98105[outcome]

model = sm.OLS(y, X)
result_98105 = model.fit()

# Get standardized residuals (convert to pandas Series)
sresiduals = pd.Series(
    result_98105.get_influence().resid_studentized_internal,
    index=house_98105.index
)

# Identify the most extreme outlier
outlier = house_98105.loc[sresiduals.idxmin()]

print('AdjSalePrice:', outlier[outcome])
print(outlier[predictors])
