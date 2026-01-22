import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Example data: four_sessions DataFrame with 'Time' and 'Page' columns
data = {
    'Page': ['Page 1']*5 + ['Page 2']*5 + ['Page 3']*5 + ['Page 4']*5,
    'Time': [164, 172, 177, 156, 195,   # Page 1
             178, 191, 182, 185, 177,   # Page 2
             175, 193, 171, 163, 176,   # Page 3
             155, 166, 164, 170, 168]   # Page 4
}
four_sessions = pd.DataFrame(data)

# ANOVA using statsmodels
model = smf.ols('Time ~ Page', data=four_sessions).fit()
aov_table = sm.stats.anova_lm(model)
print('ANOVA table:')
print(aov_table)
