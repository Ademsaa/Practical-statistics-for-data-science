from scipy import stats

# Parameters
n = 5      # number of trials
p = 0.1    # probability of success

# Probability of exactly 2 successes
prob_exact = stats.binom.pmf(2, n=n, p=p)

# Probability of 2 or fewer successes
prob_cumulative = stats.binom.cdf(2, n=n, p=p)

print("P(X = 2) =", prob_exact)
print("P(X ≤ 2) =", prob_cumulative)
