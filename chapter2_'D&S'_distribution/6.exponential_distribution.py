from scipy import stats

# Generate 100 times between events for rate λ=2
exp_sample = stats.expon.rvs(scale=1/2, size=100)  # scale = 1/rate

print(exp_sample)