import pandas as pd
import matplotlib.pyplot as plt

# File names for Scenario S3
file_names_s3_trend = [
    "S3_n_400_Lambda_2_trend_max_sq.csv",
    "S3_n_800_Lambda_2_trend_max_sq.csv",
    "S3_n_1600_Lambda_2_trend_max_sq.csv"
]
file_names_s3_spline = [
    "S3_n_400_Lambda_2_spline_max_sq.csv",
    "S3_n_800_Lambda_2_spline_max_sq.csv",
    "S3_n_1600_Lambda_2_spline_max_sq.csv"
]

# File names for Scenario S4
file_names_s4_trend = [
    "S4_n_400_Lambda_2_trend_max_sq.csv",
    "S4_n_800_Lambda_2_trend_max_sq.csv",
    "S4_n_1600_Lambda_2_trend_max_sq.csv"
]
file_names_s4_spline = [
    "S4_n_400_Lambda_2_spline_max_sq.csv",
    "S4_n_800_Lambda_2_spline_max_sq.csv",
    "S4_n_1600_Lambda_2_spline_max_sq.csv"
]

# Initialize dictionaries to store data
s3_trend_data = {}
s3_spline_data = {}
s4_trend_data = {}
s4_spline_data = {}

# Load data for Scenario S3
for file_name in file_names_s3_trend:
    sample_size = file_name.split('_')[2]
    df = pd.read_csv(file_name)
    s3_trend_data[int(sample_size)] = df['x']

for file_name in file_names_s3_spline:
    sample_size = file_name.split('_')[2]
    df = pd.read_csv(file_name)
    s3_spline_data[int(sample_size)] = df['x']

# Load data for Scenario S4
for file_name in file_names_s4_trend:
    sample_size = file_name.split('_')[2]
    df = pd.read_csv(file_name)
    s4_trend_data[int(sample_size)] = df['x']

for file_name in file_names_s4_spline:
    sample_size = file_name.split('_')[2]
    df = pd.read_csv(file_name)
    s4_spline_data[int(sample_size)] = df['x']

# Combine data into a single list for plotting
plot_data = []
labels = []
colors = []

# Convert RGB color (100, 1.0, 10) to normalized format
custom_color = (100/255, 1.0/255, 10/255)
for sample_size in sorted(s3_trend_data.keys()):
    # Add S3 data
    plot_data.append(s3_trend_data[sample_size])
    plot_data.append(s3_spline_data[sample_size])
    labels.append(f'n={sample_size}\nS3-UnifDR')
    labels.append(f'n={sample_size}\nS3-ASS')
    colors.append('cyan')  # Color for S3-UnifDR
    colors.append('magenta')  # Color for S3-ASS

    # Add S4 data
    plot_data.append(s4_trend_data[sample_size])
    plot_data.append(s4_spline_data[sample_size])
    labels.append(f'n={sample_size}\nS4-UnifDR')
    labels.append(f'n={sample_size}\nS4-ASS')
    colors.append('indigo')  # Color for S4-UnifDR
    colors.append('red')  # Color for S4-ASS

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 10))

# Create the box plot with customized line widths
box = ax.boxplot(
    plot_data, 
    patch_artist=True, 
    flierprops={
        'marker': 'o', 
        'color': 'red', 
        'markerfacecolor': 'pink'
    },
    medianprops={
        'color': 'yellow', 
        'linewidth': 1.  # Makes the median line thicker and more notable
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
grouped_labels = [f'n={size}' for size in sorted(s3_trend_data.keys())]
ax.set_xticks([2.5 + 4 * i for i in range(len(grouped_labels))])
ax.set_xticklabels(grouped_labels, fontsize=27)

# Add a legend for the colors
legend_handles = [
    plt.Line2D([0], [0], color='cyan', lw=20, label=r'${\bf{S3}}$-UnifDR'),
    plt.Line2D([0], [0], color='magenta', lw=20, label=r'${\bf{S3}}$-ASS'),
    plt.Line2D([0], [0], color='indigo', lw=20, label=r'${\bf{S4}}$-UnifDR'),
    plt.Line2D([0], [0], color='red', lw=20, label=r'${\bf{S4}}$-ASS')
]
ax.legend(
    handles=legend_handles, 
    fontsize=30, 
    loc='upper right', 
    title='Competitors', 
    title_fontsize=30,
    bbox_to_anchor=(0.95, 1.0)  # Adjust x and y for fine-tuning the position
)

# Add title and labels with updated size and LaTeX formatting
ax.set_title(r'MSD results for Scenarios ${\bf{S3}}$ and ${\bf{S4}}$ in $\Lambda_2$', fontsize=45)
ax.set_xlabel('Sample Size', fontsize=40)
ax.set_ylabel('MSD', fontsize=40)

# Customize tick labels
ax.tick_params(axis='x', labelsize=38)
ax.tick_params(axis='y', labelsize=38)

# Display the plot
plt.tight_layout()
plt.show()


