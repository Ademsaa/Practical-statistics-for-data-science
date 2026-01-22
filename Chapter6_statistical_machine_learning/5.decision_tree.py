# -------------------------------
# 1. Import required libraries
# -------------------------------
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# -------------------------------
# 2. Define predictors and outcome
# -------------------------------

loan3000 = pd.read_csv("loan3000.csv")
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

# -------------------------------
# 3. Create the Decision Tree
# -------------------------------
loan_tree = DecisionTreeClassifier(
    random_state=1,          # for reproducibility
    criterion='entropy',     # information gain
    min_impurity_decrease=0.003  # stopping rule to prevent overfitting
)

# -------------------------------
# 4. Fit the model
# -------------------------------
loan_tree.fit(X, y)

# -------------------------------
# 5. Plot the Decision Tree
# -------------------------------
plt.figure(figsize=(12, 6))  # set figure size

plot_tree(
    loan_tree, 
    feature_names=predictors, 
    class_names=loan_tree.classes_, 
    filled=True,        # color nodes by class
    rounded=True,       # rounded boxes
    fontsize=10
)

plt.show()
