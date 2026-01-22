# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm  # needed for GLM

# Load your dataset
loan_data = pd.read_csv('loan_data.csv')

# Ensure the outcome is categorical
loan_data['outcome'] = pd.Categorical(loan_data['outcome'])

# Define predictors and outcome
predictors = ['payment_inc_ratio', 'purpose_', 'home_', 'emp_len_', 'borrower_score']
outcome = 'outcome'

# Create dummy variables for categorical predictors
X = pd.get_dummies(loan_data[predictors], drop_first=True)

# Convert all columns to numeric (float)
X = X.astype(float)

y = loan_data[outcome]

# Initialize and fit logistic regression model (scikit-learn)
logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
logit_reg.fit(X, y)

# Convert categorical outcome to numeric for statsmodels
y_numbers = [1 if yi == 'default' else 0 for yi in y]

# Fit logistic regression using statsmodels
logit_reg_sm = sm.GLM(y_numbers, X.assign(const=1),
                      family=sm.families.Binomial())
logit_result = logit_reg_sm.fit()

# Show summary
print(logit_result.summary())

import statsmodels.formula.api as smf
formula = ('outcome ~ bs(payment_inc_ratio, df=4) + purpose_ + ' +
'home_ + emp_len_ + bs(borrower_score, df=4)')
model = smf.glm(formula=formula, data=loan_data, family=sm.families.Binomial())
results = model.fit()
print(results.summary())