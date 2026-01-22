from scipy import stats
import numpy as np

# Contingency table: rows = headlines, columns = clicks / no-clicks
clicks = np.array([
    [14, 986],  # Headline A
    [8, 992],   # Headline B
    [12, 988]   # Headline C
])

# Perform chi-square test
chisq, pvalue, df, expected = stats.chi2_contingency(clicks)

# Print results
print(f'Observed chi2: {chisq:.4f}')
print(f'p-value: {pvalue:.4f}')
print(f'Degrees of freedom: {df}')
print('Expected counts:\n', expected)
