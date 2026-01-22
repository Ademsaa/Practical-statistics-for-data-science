import random
import pandas as pd
import matplotlib.pyplot as plt

# Define permutation function
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.iloc[list(idx_B)].mean() - x.iloc[list(idx_A)].mean()

# Observed percentage difference
obs_pct_diff = 100 * (200 / 23739 - 182 / 22588)
print(f'Observed difference: {obs_pct_diff:.4f}%')

# Create conversion data
conversion = [0] * 45945
conversion.extend([1] * 382)
conversion = pd.Series(conversion)

# Run permutation test
perm_diffs = [100 * perm_fun(conversion, 23739, 22588) for _ in range(1000)]

# Plot histogram
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=obs_pct_diff, color='black', lw=2)
ax.text(0.06, 200, 'Observed\ndifference', bbox={'facecolor':'white'})
ax.set_xlabel('Conversion rate (percent)')
ax.set_ylabel('Frequency')
plt.show()

#================================================
import numpy as np
p_value = np.mean([diff > obs_pct_diff for diff in perm_diffs])
print("p_value = ", p_value)

#=================================================
import numpy as np
from scipy import stats

# Data: [successes, failures] for each group
survivors = np.array([[200, 23739 - 200], 
                      [182, 22588 - 182]])

# Perform chi-square test
chi2, p_value, df, _ = stats.chi2_contingency(survivors)

#
print(f'p-value: {p_value / 2:.4f}')
