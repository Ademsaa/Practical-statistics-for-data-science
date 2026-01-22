import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Parameters
shape = 1.5        # beta
scale = 5000       # eta (characteristic life)
size = 100         # number of random values

# Generate random lifetimes
weibull_lifetimes = stats.weibull_min.rvs(c=shape, scale=scale, size=size)

# Print first 10 values
print(weibull_lifetimes[:10])

# Optional: visualize with a histogram
plt.hist(weibull_lifetimes, bins=20, edgecolor='k')
plt.xlabel('Lifetime')
plt.ylabel('Frequency')
plt.title('Weibull Distribution (shape=1.5, scale=5000)')
plt.show()
