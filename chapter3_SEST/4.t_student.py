import pandas as pd
from scipy import stats

# Load your session times data
session_times = pd.read_csv("web_page_data.csv")

# Perform t-test (Welch’s t-test: unequal variances)
res = stats.ttest_ind(
    session_times[session_times.Page == 'Page A'].Time,
    session_times[session_times.Page == 'Page B'].Time,
    equal_var=False
)

# Print t-statistic and single-sided p-value
print(f't-statistic: {res.statistic:.4f}')
print(f'p-value for single sided test: {res.pvalue / 2:.4f}')
