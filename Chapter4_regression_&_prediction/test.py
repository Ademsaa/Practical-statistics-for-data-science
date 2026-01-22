import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

house = pd.read_csv('house_sales.csv', sep = '\t')

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

house_lm = LinearRegression()
house_lm.fit(house[predictors], house[outcome])

print(f"intercept: {house_lm.intercept_:.3f}")
coeficients  = house_lm.coef_ #this is a list
print("Model's coeficents: ")
for name, coef in zip(predictors, coeficients):
    print(f"{name}:{coef}")

#====================================================================

#calculate RMSE and R2 squared
fitted_values = house_lm.predict(house[predictors])
rmse = np.sqrt(mean_squared_error(house[outcome], fitted_values))
r2 = r2_score(house['AdjSalePrice'], fitted_values)

print(f"RMSE: {rmse:.2f}")
print(f"r2: {r2:.4f}")

X = house[predictors].assign(const=1)
y = house[outcome]
#=====================================================================

#Fit OLS model
x = house[predictors].assign(const = 1)
y = house[outcome]

model = sm.OLS(y,x)
results = model.fit()
print(results.summary())
