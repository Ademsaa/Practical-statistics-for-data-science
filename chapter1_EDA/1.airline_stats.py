import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data directly from the CSV file
# This assumes "1.airline_stats.csv" is in the current working directory.
airline_stats = pd.read_csv("1.airline_stats.csv")

# 2. Generate the boxplot using the pandas DataFrame method
# 'by' groups the data by the specified column ('airline')
# 'column' selects the variable to plot ('pct_carrier_delay')
ax = airline_stats.boxplot(
    by='airline', 
    column='pct_carrier_delay', 
    figsize=(8, 6),
)

# 3. CORRECT WAY to set the Y-axis limit using the Axes object:
ax.set_ylim(0, 50) 

# 4. Apply customizations
ax.set_xlabel('')                           # Removes the default x-axis label ('airline')
ax.set_ylabel('Daily % of Delayed Flights') # Sets the new descriptive y-axis label
plt.suptitle('')                            # Removes the automatic pandas/matplotlib super-title

# 5. Display the plot
plt.show()