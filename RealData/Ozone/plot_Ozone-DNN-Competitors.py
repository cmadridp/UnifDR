import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load the data
plot_data_unifdr = pd.read_csv('plot_ozone_dnn.csv', engine='python')  # Renamed from 'DNN'
plot_data_cart = pd.read_csv('plot_ozone_cart.csv', engine='python')
plot_data_mars = pd.read_csv('plot_ozone_mars.csv', engine='python')
plot_data_rf = pd.read_csv('plot_ozone_rf.csv', engine='python')
plot_data_drf = pd.read_csv('plot_ozone_drf.csv', engine='python')
plot_data_engression = pd.read_csv('plot_ozone_engression.csv', engine='python')

# Debugging step: Check Random Forest data range
print("Random Forest Data Summary:")
print(plot_data_rf[['Longitude', 'Latitude']].describe())

# Combine the datasets
plot_data_combined = pd.concat([
    plot_data_unifdr, plot_data_cart, plot_data_mars, plot_data_rf, plot_data_drf, plot_data_engression
], ignore_index=True)

# Exclude Alaska (Longitude < -130 and Latitude > 50)
plot_data_combined = plot_data_combined[
    ~((plot_data_combined['Longitude'] < -130) & (plot_data_combined['Latitude'] > 50))
]

# Exclude Hawaii (Longitude between -161 and -154, Latitude between 18 and 23)
plot_data_combined = plot_data_combined[
    ~((plot_data_combined['Longitude'].between(-161, -154)) & 
      (plot_data_combined['Latitude'].between(18, 23)))
]

# Define the new order of methods
methods = ['DNN', 'CART', 'MARS', 'Random Forest', 'DRF', 'Engression']
titles = ['UnifDR', 'CART', 'MARS', 'Random Forest', 'DRF', 'Engression']

# Set a global color scale for fair comparison across methods
vmin_global = plot_data_combined['Prediction'].min()
vmax_global = plot_data_combined['Prediction'].max()

# Create the plot with multiple subplots
fig, axes = plt.subplots(
    2, 3, figsize=(18, 9), subplot_kw={'projection': ccrs.PlateCarree()}, 
    gridspec_kw={'wspace': 0.02, 'hspace': -0.05}  # Reduce space between rows
)

axes = axes.flatten()

for i, method in enumerate(methods):
    ax = axes[i]
    
    # Add geographical features
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
    ax.add_feature(cfeature.STATES, edgecolor='gray', linestyle='--')

    # Plot ozone concentration data
    subset = plot_data_combined[plot_data_combined['Method'] == method]
    
    scatter = ax.scatter(
        subset['Longitude'], subset['Latitude'], 
        c=subset['Prediction'], cmap='coolwarm', s=40, 
        edgecolors='black', linewidths=0.5, alpha=0.9,  # Ensures visibility and proper contrast
        transform=ccrs.PlateCarree(), vmin=vmin_global, vmax=vmax_global  # Use global color scale
    )

    # Zoom into the continental U.S. and Puerto Rico
    ax.set_xlim([-126, -64])  # Longitude limits
    ax.set_ylim([16.5, 50])  # Latitude limits: Canada-U.S. border at top, Puerto Rico at bottom
    
    ax.set_xlabel('Longitude', fontsize=18)
    ax.set_ylabel('Latitude', fontsize=18)

    # Reduce space between subplot titles and main title
    ax.set_title(titles[i], fontsize=25, fontweight='bold', color='blue', pad=0.5)

# Add a single colorbar for all plots (ensuring global scale)
cbar_ax = fig.add_axes([0.2, 0.09, 0.6, 0.04])  # Shift colorbar slightly upward
cbar = fig.colorbar(scatter, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Ozone Concentration', fontsize=25, labelpad=1)
cbar.ax.tick_params(labelsize=25, pad=0.1)

# Keep title position unchanged
plt.suptitle("Predictions for Ozone Concentration CDF in the USA (Dense ReLU Networks)", 
             fontsize=28, fontweight='bold', y=0.99)

# Shift everything **below the title UP** without moving the title
plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.15)  # Reduce `bottom` to shift content up

plt.show()
