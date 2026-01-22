import pandas as pd
import mca
import matplotlib.pyplot as plt

# Load the dataset
housetasks = pd.read_csv("housetasks.csv", index_col=0)

# Perform MCA (acts like CA for counts)
ca = mca.MCA(housetasks, ncols=housetasks.shape[1])

# Row coordinates (tasks)
rows_2d = ca.fs_r(N=2)

# Column coordinates (actors, e.g., Wife/Husband)
cols_2d = ca.fs_c(N=2)

# Percentage of variance explained
dim1_var = ca.expl_var(N=1)[0] * 100
dim2_var = ca.expl_var(N=2)[1] * 100

# Plot
plt.figure(figsize=(7,7))

# Plot rows (tasks) as blue dots
plt.scatter(rows_2d[:,0], rows_2d[:,1], color='blue', s=80, label='Tasks')
for i, task in enumerate(housetasks.index):
    plt.text(rows_2d[i,0]+0.02, rows_2d[i,1]+0.02, task, color='blue', fontsize=9)

# Plot columns (actors) as red triangles
plt.scatter(cols_2d[:,0], cols_2d[:,1], color='red', marker='^', s=100, label='Actors')
for i, actor in enumerate(housetasks.columns):
    plt.text(cols_2d[i,0]+0.02, cols_2d[i,1]+0.02, actor, color='red', fontsize=9)

# Bold axes at 0
plt.axhline(0, color='black', linewidth=2)
plt.axvline(0, color='black', linewidth=2)

# Grid for contrast
plt.grid(True, linestyle='--', alpha=0.5)

# Labels with variance percentages
plt.xlabel(f"Dimension 1 ({dim1_var:.1f}%)")
plt.ylabel(f"Dimension 2 ({dim2_var:.1f}%)")
plt.title("Correspondence Analysis Biplot")

plt.legend()
plt.tight_layout()
plt.show()
