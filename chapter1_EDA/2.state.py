import pandas as pd
from  statsmodels import robust


state = pd.read_csv("2.state.csv")

#1.standard deviation
standrad_deviation = state['Population'].std()

#IQR
iqr = state['Population'].quantile(0.75) - state['Population'].quantile(0.25)

#Median absolute deviation
mad = robust.scale.mad(state['Population'])

print("standrad deviation: ", standrad_deviation)
print("iqr: ", iqr)
print("mad: ", mad)