import statsmodels.api as sm

# Baseline and improved click-through rates
p1 = 0.011    # 1.1%
p2 = 0.0121   # 1.21%

# Compute effect size for proportions
effect_size = sm.stats.proportion_effectsize(p2, p1)

# Power analysis for two independent samples
analysis = sm.stats.TTestIndPower()

# Solve for required sample size per group
sample_size = analysis.solve_power(
    effect_size=effect_size,
    alpha=0.05,
    power=0.8,
    alternative='larger'
)

print(f"Sample Size per group: {sample_size:.0f}")
