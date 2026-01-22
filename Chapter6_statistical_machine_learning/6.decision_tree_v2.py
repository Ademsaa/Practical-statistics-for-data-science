# -------------------------------
# 1. Import required libraries
# -------------------------------
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# 2. Load your dataset
# -------------------------------
# Replace with your dataset
loan3000 = pd.read_csv("loan3000.csv")

# -------------------------------
# 3. Define predictors and outcome
# -------------------------------
predictors = ['borrower_score', 'payment_inc_ratio']
outcome = 'outcome'

X = loan3000[predictors]
y = loan3000[outcome]

# -------------------------------
# 4. Train the Decision Tree
# -------------------------------
loan_tree = DecisionTreeClassifier(
    random_state=1,
    criterion='entropy',
    min_impurity_decrease=0.003
)

loan_tree.fit(X, y)

# -------------------------------
# 5. Function to print tree node-by-node
# -------------------------------
def textDecisionTree(tree, feature_names=None):
    tree_ = tree.tree_
    # Use feature names if provided
    feature_name = [feature_names[i] if i != -2 else "undefined!" for i in tree_.feature]

    def recurse(node):
        if tree_.feature[node] != -2:  # test node
            print(f"node={node} test node: go to node {tree_.children_left[node]} "
                  f"if {feature_name[node]} <= {tree_.threshold[node]} "
                  f"else to node {tree_.children_right[node]}")
            recurse(tree_.children_left[node])
            recurse(tree_.children_right[node])
        else:  # leaf node
            probs = tree_.value[node][0] / np.sum(tree_.value[node][0])
            print(f"node={node} leaf node: {probs.tolist()}")

    recurse(0)

# -------------------------------
# 6. Print the decision tree
# -------------------------------
textDecisionTree(loan_tree, feature_names=predictors)
