import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
kc_tax = pd.read_csv("1.kc_tax.csv")

# 2. Filter the data
kc_tax0 = kc_tax.loc[
    (kc_tax.TaxAssessedValue < 750000) &
    (kc_tax.SqFtTotLiving > 100) &
    (kc_tax.SqFtTotLiving < 3500),
    :
]

# 3. Figure
plt.figure(figsize=(12, 6))

# 4. Scatterplot of points (purple, transparent)
plt.scatter(
    kc_tax0.SqFtTotLiving,
    kc_tax0.TaxAssessedValue,
    s=10,
    color=(0.6, 0, 1, 0.25),  # RGBA: purple with alpha
    edgecolor=None
)

# 5. Labels
plt.xlabel("Finished Square Feet", fontsize=12)
plt.ylabel("Tax-Assessed Value", fontsize=12)
plt.title("Scatter + White KDE Contours", fontsize=14, fontweight="bold")

plt.show()

# 5. KDE contour lines (white)
plt.figure(figsize=(12, 6))

sns.kdeplot(
    x=kc_tax0.SqFtTotLiving,
    y=kc_tax0.TaxAssessedValue,
    levels=10,
    color="black",
    linewidths=1.2
)

plt.xlabel("Finished Square Feet", fontsize=12)
plt.ylabel("Tax-Assessed Value", fontsize=12)
plt.title("KDE Contour Plot", fontsize=14, fontweight="bold")

plt.show()   