import pandas as pd
import matplotlib.pyplot as plt

# File names for Scenario S5
file_names_s5 = [
    "S5_n_400_Lambda_1_cart_crps.csv", "S5_n_400_Lambda_1_dnn_crps.csv",
    "S5_n_400_Lambda_1_drf_crps.csv", "S5_n_400_Lambda_1_engression_crps.csv",
    "S5_n_400_Lambda_1_mars_crps.csv", "S5_n_400_Lambda_1_rf_crps.csv",
    "S5_n_800_Lambda_1_cart_crps.csv", "S5_n_800_Lambda_1_dnn_crps.csv",
    "S5_n_800_Lambda_1_drf_crps.csv", "S5_n_800_Lambda_1_engression_crps.csv",
    "S5_n_800_Lambda_1_mars_crps.csv", "S5_n_800_Lambda_1_rf_crps.csv",
    "S5_n_1600_Lambda_1_cart_crps.csv", "S5_n_1600_Lambda_1_dnn_crps.csv",
    "S5_n_1600_Lambda_1_drf_crps.csv", "S5_n_1600_Lambda_1_engression_crps.csv",
    "S5_n_1600_Lambda_1_mars_crps.csv", "S5_n_1600_Lambda_1_rf_crps.csv"
]

# Initialize dictionary to store data
s5_data = {}

# Load data for Scenario S5
for file_name in file_names_s5:
    parts = file_name.split('_')
    sample_size = parts[2]  # Correct extraction of sample size
    method = parts[5].replace('.csv', '').strip().lower()  # Extract the correct method
    
    # Read data skipping the first row and assuming a single-column CSV without a header
    df = pd.read_csv(file_name, skiprows=1, header=None, names=['x'])
    
    if int(sample_size) not in s5_data:
        s5_data[int(sample_size)] = {}

    s5_data[int(sample_size)][method] = df['x']

    # Debug print to check extraction
    print(f"Processing file: {file_name}, Extracted method: {method}, Available methods: {s5_data[int(sample_size)].keys()}")


# Combine data into a single list for plotting
plot_data = []
labels = []
colors = []

# Define colors for each method
method_colors = {
    "cart": "tan",
    "dnn": "lightcoral",
    "drf": "gold",
    "engression": "yellowgreen",
    "mars": "cornflowerblue",
    "rf": "mediumturquoise"
}

# Prepare data for box plot
for sample_size in sorted(s5_data.keys()):
    for method in method_colors.keys():
        if method in s5_data[sample_size]:
            plot_data.append(s5_data[sample_size][method])
            labels.append(f'n={sample_size}\n{method.upper()}')
            colors.append(method_colors[method])
        else:
            print(f"Warning: Method {method} not found for n={sample_size}")

# Create the figure and axis
fig, ax = plt.subplots(figsize=(16, 10))

# Create the box plot with customized line widths
box = ax.boxplot(
    plot_data,
    patch_artist=True,
    flierprops={'marker': 'o', 'color': 'red', 'markerfacecolor': 'silver'},
    medianprops={'color': 'black', 'linewidth': 1.5},
    boxprops={'linewidth': 2.5},
    whiskerprops={'linewidth': 2.5},
    capprops={'linewidth': 2.5}
)

# Apply colors to the boxes
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Customize x-axis labels to group by sample size
grouped_labels = [f'n={size}' for size in sorted(s5_data.keys())]
ax.set_xticks([3.5 + 6 * i for i in range(len(grouped_labels))])
ax.set_xticklabels(grouped_labels, fontsize=27)

# Add a legend for the colors
legend_handles = [
    plt.Line2D([0], [0], color=method_colors["cart"], lw=20, label='CART'),
    plt.Line2D([0], [0], color=method_colors["dnn"], lw=20, label='UnifDR'),
    plt.Line2D([0], [0], color=method_colors["drf"], lw=20, label='DRF'),
    plt.Line2D([0], [0], color=method_colors["engression"], lw=20, label='EnG'),
    plt.Line2D([0], [0], color=method_colors["mars"], lw=20, label='MARS'),
    plt.Line2D([0], [0], color=method_colors["rf"], lw=20, label='RF')
]
ax.legend(
    handles=legend_handles,
    fontsize=23.5,
    loc='upper right',
    title='Competitors',
    title_fontsize=23.5,
    bbox_to_anchor=(0.85, 1.02)
)

# Add title and labels with updated size and LaTeX formatting
ax.set_title(r'CRPS results for Scenario ${\bf{S5}}$ in $\Lambda_1$', fontsize=45)
ax.set_xlabel('Sample Size', fontsize=40)
ax.set_ylabel('CRPS', fontsize=40)

# Customize tick labels
ax.tick_params(axis='x', labelsize=38)
ax.tick_params(axis='y', labelsize=38)

# Display the plot
plt.tight_layout()
plt.show()
