import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data directly from the CSV file
# This assumes "1.airline_stats.csv" is in the current working directory.
airline_stats = pd.read_csv("1.airline_stats.csv")

# 2. Generate the violin plot using the modern seaborn syntax
# x='airline' defines the grouping variable (categories)
# y='pct_carrier_delay' defines the distribution variable
plt.figure(figsize=(10, 6)) # Define the figure size first
ax = sns.violinplot(
    data=airline_stats, 
    x='airline', 
    y='pct_carrier_delay',
    inner='quartile', 
    color='skyblue' # Using a pleasant color, but 'white' might be problematic on a white background
)

# 3. Apply customizations
ax.set_xlabel('')                           # Removes the x-axis label ('airline')
ax.set_ylabel('Daily % of Delayed Flights') # Sets the new descriptive y-axis label
ax.set_title('Distribution of Carrier-Caused Delays by Airline') # Adding a proper title

# 4. Set the Y-axis limit for consistency (0 to 50)
ax.set_ylim(0, 50) 

# 5. Display the plot
plt.show()