import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
house = pd.read_csv('house_sales.csv', sep='\t')

# Show the first few values of the PropertyType column
property_type = house['PropertyType']
print("First 5 PropertyType values:")
print(property_type.head())

# Convert PropertyType to dummy variables (one-hot encoding)
property_dummies = pd.get_dummies(house['PropertyType']) 
print("\nOne-hot encoded PropertyType (all columns):")
print(property_dummies.head())

# Convert PropertyType to dummy variables and drop the first column
property_dummies_drop = pd.get_dummies(house['PropertyType'], drop_first=True)
print("\nrefrence encoded PropertyType (drop first column):")
print(property_dummies_drop.head())