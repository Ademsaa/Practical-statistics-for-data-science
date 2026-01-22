import pandas as pd
from scipy.stats import trim_mean
import numpy as np
import wquantiles as wq
from  statsmodels import robust
import matplotlib.pyplot as plt


store = pd.read_csv('store.csv', encoding = "Windows-1252")


print("\n#============= Estimate of location ==================") 

mean_sales = store["Sales"].mean()
print(f"-Mean of 'Sales': {mean_sales: .2f}")

median_sales = store["Sales"].median()
print(f"-Median of 'Sales': {median_sales: .2f}")

trimmed_mean_sales = trim_mean(store["Sales"], 0.1) 
print(f"-Trimmed mean of 'Sales': {trimmed_mean_sales: .2f}")

weighted_mean = np.average(store["Sales"], weights=store["Quantity"])
print(f"-Weighted mean of 'Sales' using 'Quantity' as weight: {weighted_mean: .2f}")

weighted_median = wq.median(store["Sales"], weights=store["Quantity"])
print(f"-Weighted median of 'Sales' using 'Quantity' as weight: {weighted_median: .2f}")




print("\n#============= Estimate of variability ==================") 
print("Calculation of deviation: ")
deviation = store["Sales"] - median_sales
print(round(deviation,2),end = "\n\n")

mad_mean = (store["Sales"] - store["Sales"].mean()).abs().mean()
print(f"-Mean absolute deviation of 'Sales': {mad_mean: .2f}")

variance = store["Sales"].var()
print(f"-Variance (the average of squared deviation from the mean) of 'Sales': {variance: .2f}")

standard_deviation = store["Sales"].std()
print(f"-Standard deviation of 'Sales': {standard_deviation: .2f}")

mad_median = robust.scale.mad(store["Sales"])
print(f"-Median absolute deviation of 'Sales': {mad_median: .2f}")

iqr = store["Sales"].quantile(0.75) - store["Sales"].quantile(0.25)
print(f"-IQR of 'Sales': {iqr: .2f}")



print("\n#============= Exploring data visualisation ==================") 


#================== Boxplot =========================

ax = (store["Sales"]).plot.box()
ax.set_ylabel("Sales($)")
plt.show()



ax = store.boxplot(
    by = "Category",
    column = "Sales",
    figsize = (8,8)
)



ax.set_xlabel('')
plt.xticks(rotation=90)
ax.set_ylabel('Toatal revenues')
plt.suptitle('')

plt.show()
 

#================== Histogram ========================

right_lim = 7500
mybins = np.arange(0, right_lim+200, 200)
print(mybins)
ax = (store["Sales"]).plot.hist(figsize = (10,10),
                                bins = mybins,
                                edgecolor = "black",
                                linewidth=0.7)

ax.set_xlim(left=0, right= right_lim)


ax.set_title("Sales ranges frequency")
ax.set_xlabel("Sales ranges")
ax.set_ylabel("Count")
ax.tick_params(labelsize = 10, color = "darkblue")
plt.show()


#=====================================================
# right_lim = 7500
# mybins = np.linspace(0, right_lim, 15, endpoint=True, dtype=int)
# print(mybins)
# ax = (store["Sales"]).plot.hist(figsize = (10,10),
#                                 bins = mybins,
#                                 edgecolor = "black",
#                                 linewidth=0.7)

# ax.set_xlim(left=0, right= right_lim)


# ax.set_title("Sales ranges frequency")
# ax.set_xlabel("Sales ranges")
# ax.set_ylabel("Count")
# ax.tick_params(labelsize = 10, color = "darkblue")
# plt.show()

#=====================================================
right_lim = 7500
mybins = np.arange(0, right_lim+200, 200)
print(mybins)
ax = (store["Sales"]).plot.hist(figsize = (10,10),
                                bins = mybins,
                                edgecolor = "black",
                                linewidth=0.7,
                                density=True)

store["Sales"].plot.density(ax=ax)

ax.set_xlim(left=0, right= right_lim)

ax.set_title("Sales ranges density")
ax.set_xlabel("Sales ranges")
ax.set_ylabel("Density")
ax.tick_params(labelsize = 10, color = "darkblue")
plt.show()

#for calculating area under the density curve
from scipy.stats import gaussian_kde

# Your range
a = 200
b = 400

# Extract Sales data
sales = store["Sales"]

# Estimate density using KDE
kde = gaussian_kde(sales)
x = np.linspace(sales.min(), sales.max(), 1000)  # grid points
y = kde(x)

# Select the region between a and b
mask = (x >= a) & (x <= b)
x_region = x[mask]
y_region = y[mask]

# Calculate area under the curve using the trapezoidal rule
area = np.trapz(y_region, x_region)

print(f"Probability that Sales is between {a} and {b}: {area:.4f}")

#plot only the density plot
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Example data
sales = store["Sales"]

# Create KDE
kde = gaussian_kde(sales)

# Create a grid of x values
x = np.linspace(sales.min(), sales.max(), 1000)
print(x)

# Evaluate KDE on this grid
y = kde(x)

# Plot
plt.figure(figsize=(10,6))
plt.plot(x, y, color='blue', lw=2)
plt.title("KDE of Sales")
plt.xlabel("Sales")
plt.ylabel("Density")
plt.show()

#=====================================================================
#barcharts
#y values
state_counts = store['State'].value_counts()
print(state_counts)

# Bar plot
ax = state_counts.plot.bar(color='skyblue', figsize=(10,6))
plt.title("Number of Sales per State")
plt.xlabel("State")
plt.ylabel("Count of Transactions")
plt.xticks(rotation = 90)
plt.show()

#====================================================================== 

