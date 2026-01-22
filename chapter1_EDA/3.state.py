import pandas as pd
from scipy.stats import trim_mean
import numpy as np
    #import wquantiles as wq

# Read CSV file
state = pd.read_csv('1.state.csv')

quantiles = state['MurderRate'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
print(quantiles)
