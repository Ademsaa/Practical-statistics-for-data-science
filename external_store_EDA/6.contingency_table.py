import pandas as pd

# Load data
store = pd.read_csv("store.csv", encoding="Windows-1252")

# Contingency table (counts)
crosstab = store.pivot_table(
    index="Segment", #rows
    columns="Category", #columns
    values="Order ID", 
    aggfunc="count", #operation
    margins=True #include All
)

print(crosstab)

# Export counts to TXT
with open("contingency_table.txt", "w", encoding="utf-8") as file:
    file.write(crosstab.to_string())
    print("file uploaded successfully")

# ---- PERCENTAGES ----

# Remove 'All' row temporarily for row-wise percentages
base = crosstab.drop(index="All") #this is pivot table witthout 'All' row
# Columns without 'All'
category_cols = base.columns.drop("All") #this line removes the 'All' column from the table 


# Row-wise percentages: P(Category | Segment)
percentage_crosstab = base[category_cols].div(base["All"], axis=0)
#base["All"] is the 'All' for each row
#div(base["All"], axis=0) means devide each row by corresponding value of 'All'
#axis=0 = to divide by 'All' column
#axis = 1 to divide by 'All' row

# Add segment share of total orders
percentage_crosstab["All"] = base["All"] / base["All"].sum()  
#devide each row 'All' by total all

# Replace 'All' row with NaN (since it has no meaningful row-wise percentage)
percentage_crosstab.loc['All'] = float('nan')
#.loc allows you to access or modify rows (or columns) by their labels.

print("\nPercentages:\n", percentage_crosstab)
