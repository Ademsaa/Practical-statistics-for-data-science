import pandas as pd
import matplotlib.pyplot as plt

# Assuming session_times is your DataFrame with columns 'Page' and 'Time'

session_times = pd.read_csv("web_page_data.csv")
# Create boxplot by page

ax = session_times.boxplot(by='Page', column='Time', figsize=(7,8))
ax.set_xlabel('')
ax.set_ylabel('Time (in seconds)')
plt.suptitle('')  # Remove the automatic title
plt.show()

# Calculate mean session time for each page
mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()

# Difference between Page B and Page A
mean_difference = mean_b - mean_a
print("Mean difference (Page B - Page A):", mean_difference)
#==================================================================
