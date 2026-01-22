import pandas as pd
import numpy as np

store = pd.read_csv("store.csv", encoding="Windows-1252")

range = np.arange(0, store["Sales"].max(), 200)
print(range)
print("\n\n =====================================================================")
linspace = np.linspace(0, 7500, num = 40, endpoint=True, dtype=int)
print(linspace)