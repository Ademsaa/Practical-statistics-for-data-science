import pandas as pd

# Read CSV

lc_loans = pd.read_csv("1.lc_loans.csv")

# Pivot table with counts
crosstab = lc_loans.pivot_table(
index='grade',
columns='status',
aggfunc=lambda x: len(x),
margins=True #this line to add All column 
)

# Keep only grades A to G
df_counts = crosstab.loc['A':'G', :].copy()

# Display separate tables
print("Counts:\n", df_counts)


#===================================================
# Calculate row percentages for status columns exept All
status_cols = df_counts.columns.drop('All')
print(status_cols)
#it does not remove the column from the DataFrame itself — 
# it just returns the list of column names without 'All'.
#Output:
# Index(['Charged Off', 'Current', 'Fully Paid', 'Late'], dtype='object', name='status')

#make an exact copy of the table
df_perc = df_counts.copy()
#devide each status by 'all' value of each row
df_perc[status_cols] = df_perc[status_cols].div(df_perc['All'], axis=0)

# Calculate propotion of each 'All' grade to total 'All'
df_perc['All'] = df_counts['All'] / df_counts['All'].sum()
print("\nPercentages:\n", df_perc)