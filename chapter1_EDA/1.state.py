import pandas as pd
from scipy.stats import trim_mean
import numpy as np
    #import wquantiles as wq

# Read CSV file
state = pd.read_csv('1.state.csv')

# Compute statistics
mean_value = state['Population'].mean()
trimmed_mean = trim_mean(state['Population'], 0.1)  # 10% trimmed mean
median_value = state['Population'].median()

# Print results
print("Mean:", mean_value)
print("Trimmed mean:", trimmed_mean)
print("Median:", median_value)

#weightned_mean and ledian
w_mean = np.average(state["MurderRate"], weights=state["Population"])
#w_median = wq.median(state["MurderRate"], weights= state["Population"])

print("weighted_mean:", w_mean)
