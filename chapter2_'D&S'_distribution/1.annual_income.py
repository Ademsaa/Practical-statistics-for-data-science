import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
loans_income = pd.read_csv('loans_income.csv')['x']  # the column is named 'x'

# Sample data
sample_data = pd.DataFrame({ #pd.DataFrame({...}) Combines the two columns into a new DataFrame called sample_data.
    'income': loans_income.sample(1000, random_state=42), #.sample(1000) randomly selects 1000 values from loans_income. 
                                                        #random_state=42 ensures that every time you run the code, 
                                                        # you get the same 1000 values (useful for reproducibility).
                                                        #This creates a column named "income" in a new DataFrame with the 1000 sampled income values.
    'type': 'Data', #This creates another column named "type" with the string 'Data' repeated 1000 times (one for each row).
                    #It is used later for labeling the plot in FacetGrid so Seaborn knows this is the "raw data" histogram.
})

# Sample means of 5
sample_mean_05 = pd.DataFrame({
    'income': [loans_income.sample(5, random_state=i).mean() for i in range(1000)],
    #This is a list comprehension that repeats something 1,000 times (for i in range(1000)).
    #Each iteration does the following:
    #loans_income.sample(5, random_state=i) → randomly selects 5 income values from the dataset.
    #random_state=i ensures different reproducible samples for each iteration.
    #.mean() → calculates the mean of those 5 incomes.
    #Result: a list of 1,000 means.
    'type': 'Mean of 5',
})

# Sample means of 20
sample_mean_20 = pd.DataFrame({
    'income': [loans_income.sample(20, random_state=i).mean() for i in range(1000)],
    'type': 'Mean of 20',
})

# Combine all data
results = pd.concat([sample_data, sample_mean_05, sample_mean_20])

# Plot using FacetGrid
g = sns.FacetGrid(results, col='type', col_wrap=1, height=4, aspect=2)
g.map_dataframe(sns.histplot, x='income', bins=40, binrange=(0, 200000))

# Set axis labels and titles
g.set_axis_labels('Income', 'Count')
g.set_titles('{col_name}')

plt.show()
