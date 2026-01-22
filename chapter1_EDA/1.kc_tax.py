import pandas as pd
import matplotlib.pyplot as plt

kc_tax = pd.read_csv("1.kc_tax.csv")

kc_tax0 = kc_tax.loc[( kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :]

print(kc_tax.shape)
print(kc_tax0.shape)

ax = kc_tax0.plot.hexbin(x='SqFtTotLiving', y='TaxAssessedValue',
                        gridsize=30, sharex=False, figsize=(5, 4), color = "purple")

ax.set_xlabel('Finished Square Feet')
ax.set_ylabel('Tax-Assessed Value')

plt.show()