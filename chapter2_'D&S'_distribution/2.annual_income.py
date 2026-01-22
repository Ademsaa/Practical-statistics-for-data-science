import pandas as pd
from sklearn.utils import resample

# Load your data
loans_income = pd.read_csv("loans_income.csv")['x']  # adjust column name if needed

# Bootstrap resampling
results = []

for nrepeat in range(1000):
    sample = resample(loans_income, replace=True)  # resample with replacement
    results.append(sample.median())

# Convert results to a pandas Series
results = pd.Series(results)

# Print bootstrap statistics
print('Bootstrap Statistics:')
print(f'Original median: {round(loans_income.median())}')
print(f'Bias: {round(results.mean() - loans_income.median(),2)}')
print(f'Standard error: {round(results.std(),2)}')
