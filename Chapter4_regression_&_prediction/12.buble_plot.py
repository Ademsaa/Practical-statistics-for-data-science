import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import matplotlib.pyplot as plt

# Load the dataset
house = pd.read_csv('house_sales.csv', sep='\t')

# Subset data for ZipCode 98105
house_98105 = house.loc[house['ZipCode'] == 98105]

# Define predictors and outcome
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# Fit OLS regression model
X = house_98105[predictors].assign(const=1)  # add intercept
y = house_98105[outcome]
model = sm.OLS(y, X)
result_98105 = model.fit()

# Compute influence measures
influence = OLSInfluence(result_98105)

# Plot influence
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(-2.5, linestyle='--', color='C1', label='±2.5 threshold')
ax.axhline(2.5, linestyle='--', color='C1')
ax.scatter(
    influence.hat_matrix_diag,  # hat values (leverage)
    influence.resid_studentized_internal,  # standardized residuals
    s=1000 * np.sqrt(influence.cooks_distance[0]),  # bubble size by Cook's distance
    alpha=0.5
)
ax.set_xlabel('Hat values (leverage)')
ax.set_ylabel('Studentized residuals')
ax.set_title('Influence Plot: ZipCode 98105')
plt.show()
