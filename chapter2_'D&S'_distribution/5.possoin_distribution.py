from scipy import stats

# Generate 100 random numbers from a Poisson distribution with λ = 2
poisson_sample = stats.poisson.rvs(mu=2, size=100)

# Print the first 10 values as an example
print(poisson_sample[:10])
