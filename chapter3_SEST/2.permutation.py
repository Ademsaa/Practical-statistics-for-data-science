import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data
session_times = pd.read_csv("web_page_data.csv")

# nA and nB are the number of sessions for Page A and Page B
nA = 21
nB = 15

# Calculate means for Page A and Page B
mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()

# Define the permutation function
def perm_fun(x, nA, nB):
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.iloc[list(idx_B)].mean() - x.iloc[list(idx_A)].mean()

# Run permutation test 1000 times
perm_diffs = [perm_fun(session_times.Time.reset_index(drop=True), nA, nB) for _ in range(1000)]

# Plot histogram
fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(perm_diffs, bins=11, rwidth=0.9)
ax.axvline(x=mean_b - mean_a, color='black', lw=2)
ax.set_xlabel('Session time differences (in seconds)')
ax.set_ylabel('Frequency')
plt.show()

# Compute permutation p-value
p_value = np.mean(np.array(perm_diffs) > (mean_b - mean_a))
print("Permutation p-value:", p_value)
