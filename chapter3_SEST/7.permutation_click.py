import random
import numpy as np

# Total clicks = 34, total impressions = 3000
box = [1] * 34
box.extend([0] * 2966)

random.shuffle(box)

# Chi-square calculation
def chi2(observed, expected):
    pearson_residuals = []
    for row, expect in zip(observed, expected):
        pearson_residuals.append(
            [(observe - expect) ** 2 / expect for observe in row]
        )
    return np.sum(pearson_residuals)

# Expected values under null hypothesis
expected_clicks = 34 / 3
expected_noclicks = 1000 - expected_clicks
expected = [expected_clicks, expected_noclicks]

# Observed clicks for headlines A, B, C
# (replace these with your real observed values)
clicks = np.array([
    [14, 8, 12],                 # observed clicks
    [984, 989, 993]              # observed no-clicks
])

chi2observed = chi2(clicks, expected)

# Permutation function
def perm_fun(box):
    sample_clicks = [
        sum(random.sample(box, 1000)),
        sum(random.sample(box, 1000)),
        sum(random.sample(box, 1000))
    ]
    sample_noclicks = [1000 - n for n in sample_clicks]
    return chi2([sample_clicks, sample_noclicks], expected)

# Run permutation test
perm_chi2 = [perm_fun(box) for _ in range(2000)]

# Compute p-value
resampled_p_value = np.mean(np.array(perm_chi2) > chi2observed)

print(f'Observed chi2: {chi2observed:.4f}')
print(f'Resampled p-value: {resampled_p_value:.4f}')
