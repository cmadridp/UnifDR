import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
plot_data_trend = pd.read_csv('plot_data_trend_C.csv', engine='python')
plot_data_spline = pd.read_csv('plot_data_spline_C.csv', engine='python')

# Add missing 'Method' column to identify the dataset source
plot_data_trend['Method'] = 'UnifDR'
plot_data_spline['Method'] = 'ASS'

# Rename columns for consistency
plot_data_trend.rename(columns={'X': 'Longitude', 'Y': 'Latitude'}, inplace=True)
plot_data_spline.rename(columns={'X': 'Longitude', 'Y': 'Latitude'}, inplace=True)

# Combine the datasets after adding the 'Method' column
plot_data_combined = pd.concat([plot_data_trend, plot_data_spline], ignore_index=True)

# Determine the global color scale for consistent colorbar
vmin = plot_data_combined['Prediction'].min()
vmax = plot_data_combined['Prediction'].max()

# Create the scatter plot with facets for each method
g = sns.FacetGrid(plot_data_combined, col="Method", height=6, aspect=1.2, sharex=True, sharey=True)
scatter = g.map_dataframe(
    sns.scatterplot, x="Longitude", y="Latitude", hue="Prediction", palette="coolwarm", edgecolor="black"
)

# Adjust axis labels and increase font size of tick labels
for ax in g.axes.flat:
    ax.set_xlabel('Longitude', fontsize=20)
    ax.set_ylabel('Latitude', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=18)  # Increase axis tick size

# Modify facet titles to remove "Method=" and change text color to blue
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(title, fontsize=24, fontweight='bold',color='blue',pad=4)

# Create a single colorbar below the plots
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Position the colorbar below the plots
cbar_ax = g.fig.add_axes([0.2, 0.12, 0.6, 0.04])  # [left, bottom, width, height]
cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Prediction", fontsize=22,labelpad=2)  # Increase font size of colorbar label
cbar.ax.tick_params(labelsize=18)  # Increase font size of colorbar ticks

# Set the title
plt.subplots_adjust(left=0.062, right=0.98, top=0.85, bottom=0.3)  # Adjust left margin

g.fig.suptitle("Predictions of the CDF of Chicago Crime Data (Trend Filtering)", fontsize=27,fontweight='bold')

plt.show()