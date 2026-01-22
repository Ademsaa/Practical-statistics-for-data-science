import matplotlib.pyplot as plt
from scipy import stats  # corrected import

# Generate figure and axis
fig, ax = plt.subplots(figsize=(4, 4))

# Generate a random normal sample
norm_sample = stats.norm.rvs(size=100)

# Create a Q-Q plot
stats.probplot(norm_sample, plot=ax)

# Show the plot
plt.show()
