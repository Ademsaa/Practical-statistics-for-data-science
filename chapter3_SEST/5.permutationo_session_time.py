import pandas as pd
import numpy as np

# Create the data
data = {
    'Page': ['Page 1']*5 + ['Page 2']*5 + ['Page 3']*5 + ['Page 4']*5,
    'Time': [164, 172, 177, 156, 195, 178, 191, 182, 185, 177,
             175, 193, 171, 163, 176, 155, 166, 164, 170, 168]
}
four_sessions = pd.DataFrame(data)

# Compute observed variance of page means
observed_variance = four_sessions.groupby('Page').mean().var()[0]
print('Observed means:', four_sessions.groupby('Page').mean().values.ravel())
print('Observed variance:', observed_variance)

# Permutation test function
def perm_test(df):
    df = df.copy()
    df['Time'] = np.random.permutation(df['Time'].values)
    return df.groupby('Page').mean().var()[0]

# Run permutation test 3000 times
perm_variance = [perm_test(four_sessions) for _ in range(3000)]

# Proportion of times permuted variance exceeds observed variance
p_value = np.mean([var > observed_variance for var in perm_variance])
print('Pr(Prob) (p-value):', p_value)
