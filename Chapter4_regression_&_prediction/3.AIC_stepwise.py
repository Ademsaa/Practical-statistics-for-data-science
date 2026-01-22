import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
house = pd.read_csv('house_sales.csv', sep='\t')

# Predictor variables
predictors = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade',
    'PropertyType', 'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt',
    'YrRenovated', 'NewConstruction'
]
# Outcome variable
outcome = 'AdjSalePrice'

# One-hot encode categorical variables
X = pd.get_dummies(house[predictors], drop_first=True)

# Ensure NewConstruction is numeric 0/1
if 'NewConstruction' in X.columns:
    X['NewConstruction'] = X['NewConstruction'].astype(int)

y = house[outcome]

# Define function to train model
def train_model(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(X[variables], y)
    return model

# Define function to score model using AIC
def score_model(model, X, y, variables):
    if len(variables) == 0:
        y_predected = [y.mean()] * len(y)
        df = 1
    else:
        y_predected = model.predict(X[variables])
        df = len(variables) + 1
    rss = ((y - y_predected) ** 2).sum()
    n = len(y)
    aic = 2 * df + n * np.log(rss / n)
    return aic

# Implement a basic forward stepwise selection
def stepwise_selection(all_variables, train_func, score_func):
    remaining = list(all_variables)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    best_model = None

    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            model = train_func(selected + [candidate])
            score = score_func(model, X, y, selected + [candidate])
            scores_with_candidates.append((score, candidate, model))


        scores_with_candidates.sort()
        best_new_score, best_candidate, best_model_candidate = scores_with_candidates[0]
        print(f"AIC of the {best_candidate}-based model = {best_new_score:.2f}")

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        best_model = best_model_candidate
        current_score = best_new_score

    return best_model, selected

# Run stepwise selection
best_model, best_variables = stepwise_selection(X.columns, train_model, score_model)

# Print results
print(f'\nIntercept: {best_model.intercept_:.3f}')
print('Coefficients:')
for name, coef in zip(best_variables, best_model.coef_):
    print(f' {name}: {coef:.3f}')
