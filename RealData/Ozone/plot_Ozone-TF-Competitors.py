import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the latest data
plot_data_trend = pd.read_csv('plot_data_trend.csv', engine='python')
plot_data_spline = pd.read_csv('plot_data_spline.csv', engine='python')

# Debug: Ensure data is actually updated
print("Updated Spline Data Preview:")
print(plot_data_spline.head())

# Ensure 'Method' column exists in both datasets
plot_data_trend["Method"] = "UnifDR"
plot_data_spline["Method"] = "ASS"

# Delete any existing combined data to ensure fresh loading
try:
    del plot_data_combined
except NameError:
    pass  # Ignore if plot_data_combined does not exist yet

# Combine the datasets after ensuring they are updated
plot_data_combined = pd.concat([plot_data_trend, plot_data_spline], ignore_index=True)

# Debug: Verify the method column is correctly assigned
print("Unique Methods in Combined Data:", plot_data_combined['Method'].unique())

# Exclude Hawaii (Longitude between -161 and -154, Latitude between 18 and 23)
plot_data_combined = plot_data_combined[
    ~((plot_data_combined['Longitude'].between(-161, -154)) & 
      (plot_data_combined['Latitude'].between(18, 23)))
]

# Define plot aesthetics
methods = ['UnifDR', 'ASS']
titles = ['UnifDR', 'ASS']

# 🔥 Compute a shared global color scale for ALL methods
vmin_global = plot_data_combined['Prediction'].min()
vmax_global = plot_data_combined['Prediction'].max()

# Create the plot with side-by-side subplots and reduced spacing
fig, axes = plt.subplots(
    1, 2, figsize=(12, 5), subplot_kw={'projection': ccrs.PlateCarree()}, gridspec_kw={'wspace': 0.02}
)

for i, method in enumerate(methods):
    ax = axes[i]
    
    # Add geographical features
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
    ax.add_feature(cfeature.STATES, edgecolor='gray', linestyle='--')

    # Filter data for current method
    subset = plot_data_combined[plot_data_combined['Method'] == method]

    # Debug: Check data is updating per method
    print(f"Plotting {method} - Data Sample:")
    print(subset.head())

    scatter = ax.scatter(
        subset['Longitude'], subset['Latitude'], 
        c=subset['Prediction'], cmap='coolwarm', s=40,  # ⬅ 🔥 Adjusted for intensity consistency
        edgecolors='black', linewidths=0.5, alpha=0.9,  # ⬅ 🔥 Keeps visibility of overlaps
        transform=ccrs.PlateCarree(), vmin=vmin_global, vmax=vmax_global  # 🔥 Use GLOBAL vmin and vmax
    )
    
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)

    # Add title labels to the top of each plot in blue color with larger font size
    ax.set_title(titles[i], fontsize=18, fontweight='bold', color='blue', pad=4)

# 🔥 Add a single colorbar with a **global scale**
cbar_ax = fig.add_axes([0.2, 0.12, 0.6, 0.04])  # [left, bottom, width, height]
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Ozone Concentration', fontsize=20, labelpad=2)  # Reduce space between bar and label
cbar.ax.tick_params(labelsize=14, pad=1)  # Adjust spacing for better readability

# Set overall title with adjusted top margin and spacing
plt.suptitle("Predictions for Ozone Concentration CDF in the USA (Trend Filtering)", fontsize=18, fontweight='bold', y=0.95)

# Adjust layout to minimize space around plots and margins
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05)

plt.show()
