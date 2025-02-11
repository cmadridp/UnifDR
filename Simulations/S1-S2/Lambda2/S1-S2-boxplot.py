import pandas as pd
import matplotlib.pyplot as plt

# File names for Scenario S1 and S2
file_names_s1 = [
    "S1_n_400_Lambda_2_crps.csv",
    "S1_n_800_Lambda_2_crps.csv",
    "S1_n_1600_Lambda_2_crps.csv"
]
file_names_s2 = [
    "S2_n_400_Lambda_2_crps.csv",
    "S2_n_800_Lambda_2_crps.csv",
    "S2_n_1600_Lambda_2_crps.csv"
]

# Initialize an empty dictionary to store data
s1_data = {}
s2_data = {}

# Load data for Scenario S1
for file_name in file_names_s1:
    sample_size = file_name.split('_')[2]  # Extract sample size from file name
    df = pd.read_csv(file_name)  # Read the CSV file
    s1_data[int(sample_size)] = df['x']  # Store the 'x' column with key as sample size

# Load data for Scenario S2
for file_name in file_names_s2:
    sample_size = file_name.split('_')[2]  # Extract sample size from file name
    df = pd.read_csv(file_name)  # Read the CSV file
    s2_data[int(sample_size)] = df['x']  # Store the 'x' column with key as sample size

# Combine S1 and S2 data into a single DataFrame for plotting
plot_data = []
labels = []
colors = []

for sample_size in sorted(s1_data.keys()):
    plot_data.append(s1_data[sample_size])
    plot_data.append(s2_data[sample_size])
    labels.append(f'n={sample_size}\nS1')
    labels.append(f'n={sample_size}\nS2')
    colors.append('blue')  # Color for S1
    colors.append('orange')  # Color for S2

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 10))

# Create the box plot with customized line widths
box = ax.boxplot(
    plot_data, 
    patch_artist=True, 
    flierprops={
        'marker': 'o', 
        'color': 'red', 
        'markerfacecolor': 'red'
    },
    medianprops={
        'color': 'yellow', 
        'linewidth': 2.5  # Makes the median line thicker and more notable
    },
    boxprops={
        'linewidth': 2.5  # Makes the box lines thicker
    },
    whiskerprops={
        'linewidth': 2.5  # Makes the whisker lines thicker
    },
    capprops={
        'linewidth': 2.5  # Makes the cap lines thicker
    }
)

# Apply colors to the boxes
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize x-axis labels to group by sample size
grouped_labels = [f'n={size}' for size in sorted(s1_data.keys())]
ax.set_xticks([1.5 + 2 * i for i in range(len(grouped_labels))])
ax.set_xticklabels(grouped_labels, fontsize=27)

# Add a legend for the colors
legend_handles = [
    plt.Line2D([0], [0], color='blue', lw=20, label=r'$\bf{S1}$'),
    plt.Line2D([0], [0], color='orange', lw=20, label=r'$\bf{S2}$')
]
ax.legend(handles=legend_handles, fontsize=30, loc='upper left', title='Scenarios', title_fontsize=30)

# Add title and labels with updated size and LaTeX formatting
ax.set_title(r'CRPS results for Scenarios ${\bf{S1}}$ and ${\bf{S2}}$ in $\Lambda_2$', fontsize=45)
ax.set_xlabel('Sample Size', fontsize=40)
ax.set_ylabel('CRPS', fontsize=40)

# Customize tick labels
ax.tick_params(axis='x', labelsize=38)
ax.tick_params(axis='y', labelsize=38)

# Display the plot
plt.tight_layout()
plt.show()
