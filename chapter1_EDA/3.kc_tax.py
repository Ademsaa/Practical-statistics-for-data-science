import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Load the data directly from the CSV file
# This assumes "1.kc_tax.csv" is in the current working directory.
try:
    kc_tax = pd.read_csv("1.kc_tax.csv")
except FileNotFoundError:
    # Providing a detailed error message if the file is missing
    print("Error: The file '1.kc_tax.csv' was not found. Please ensure it is in the correct directory.")
    exit()

# 2. Filter the data to remove extreme outliers (Equivalent to kc_tax0 in R)
kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :].copy()

# 3. Filter data for specific zip codes
zip_codes = [98188, 98105, 98108, 98126]
# Use .copy() to ensure we are working on a distinct subset
kc_tax_zip = kc_tax0.loc[kc_tax0.ZipCode.isin(zip_codes),:].copy()

# 4. Define the custom hexbin plotting function for use with FacetGrid
def hexbin(x, y, color, **kwargs):
    """
    Custom function to plot hexagonal bins using a seaborn-like color map.
    """
    # Create a light-to-dark blue color map for the gradient
    cmap = sns.light_palette('blue', as_cmap=True) 
    # Use plt.hexbin (matplotlib) to draw the plot
    plt.hexbin(x, y, gridsize=25, cmap=cmap, **kwargs)

# 5. Create the FacetGrid
g = sns.FacetGrid(kc_tax_zip, col='ZipCode', col_wrap=2, height=4, aspect=1.2)

# 6. Map the hexbin function to the data
g.map(hexbin, 'SqFtTotLiving', 'TaxAssessedValue',
      # Set the consistent x and y limits (extent) across all facets
      extent=[0, 3500, 0, 750000])

# 7. Apply labels and titles
g.set_axis_labels('Finished Square Feet', 'Tax-Assessed Value')
g.set_titles('Zip Code {col_name:.0f}')

# 8. ADDED CODE: Add grid lines to each subplot for better readability
for ax in g.axes.flat:
    # Adding a light, dashed grid
    ax.grid(True, linestyle='--', alpha=0.5, color='gray') 

# 9. Adjust layout and display the plot
plt.tight_layout()
plt.show() 