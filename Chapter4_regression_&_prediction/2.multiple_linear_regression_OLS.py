import pandas as pd
import statsmodels.api as sm

# Load dataset
house = pd.read_csv('house_sales.csv', sep='\t')

# Strip column names of spaces
house.columns = house.columns.str.strip()

# Define predictors and outcome
predictors = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade',
    'PropertyType', 'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt',
    'YrRenovated', 'NewConstruction'
]
outcome = 'AdjSalePrice'

# Keep only relevant columns
data = house[predictors + [outcome]].copy()

# Convert outcome to numeric
data[outcome] = pd.to_numeric(data[outcome], errors='coerce')

# Drop rows with missing outcome
data = data.dropna(subset=[outcome])

# One-hot encode categorical variables
X = pd.get_dummies(data[predictors], drop_first=True)

# Convert booleans to integers (important for statsmodels)
X = X.astype(float)

# Add intercept
X = sm.add_constant(X)

# Outcome variable
y = data[outcome]

# Fit OLS regression
model = sm.OLS(y, X)
results = model.fit()

# Show regression summary
print(results.summary())
