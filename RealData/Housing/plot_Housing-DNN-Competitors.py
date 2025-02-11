import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the directory where the CSV files are stored
data_dir = "/Users/carlos/Desktop/Housing"

# Load all the saved plot data
plot_data_dnn = pd.read_csv(os.path.join(data_dir, "plot_housing_dnn.csv"))
plot_data_cart = pd.read_csv(os.path.join(data_dir, "plot_housing_cart.csv"))
plot_data_mars = pd.read_csv(os.path.join(data_dir, "plot_housing_mars.csv"))
plot_data_rf = pd.read_csv(os.path.join(data_dir, "plot_housing_rf.csv"))
plot_data_drf = pd.read_csv(os.path.join(data_dir, "plot_housing_drf.csv"))
plot_data_engression = pd.read_csv(os.path.join(data_dir, "plot_housing_engression.csv"))

# Assign method names for identification
plot_data_dnn["Method"] = "UnifDR"        # DNN-based method
plot_data_cart["Method"] = "CART"
plot_data_mars["Method"] = "MARS"
plot_data_rf["Method"] = "Random Forest"
plot_data_drf["Method"] = "DRF"
plot_data_engression["Method"] = "Engression"

# Rename columns for consistency
for df in [plot_data_dnn, plot_data_cart, plot_data_mars, plot_data_rf, plot_data_drf, plot_data_engression]:
    df.rename(columns={"X": "Longitude", "Y": "Latitude"}, inplace=True)

# Combine all datasets
plot_data_combined = pd.concat(
    [plot_data_dnn, plot_data_cart, plot_data_mars, plot_data_rf, plot_data_drf, plot_data_engression],
    ignore_index=True
)

# Determine the global color scale
vmin = plot_data_combined["Prediction"].min()
vmax = plot_data_combined["Prediction"].max()

# Create facet grid for visualization
g = sns.FacetGrid(plot_data_combined, col="Method", col_wrap=3, height=5, aspect=1.2, sharex=True, sharey=True)
scatter = g.map_dataframe(
    sns.scatterplot, x="Longitude", y="Latitude", hue="Prediction", palette="coolwarm", edgecolor="black"
)

# Adjust axis labels and increase font size
for ax in g.axes.flat:
    ax.set_xlabel("Longitude", fontsize=20)
    ax.set_ylabel("Latitude", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=14)

# Modify facet titles
for ax, title in zip(g.axes.flat, g.col_names):
    ax.set_title(title, fontsize=24, fontweight='bold', color="blue", pad=4)

# Create a single colorbar below the plots
norm = plt.Normalize(vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Position the colorbar below the plots
cbar_ax = g.fig.add_axes([0.2, 0.08, 0.6, 0.04])  # [left, bottom, width, height]
cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Prediction", fontsize=22)
cbar.ax.tick_params(labelsize=18)

# Set the title
plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.2, hspace=0.2)
g.fig.suptitle("Predictions of the CDF of California Housing Prices (Dense ReLU Networks)", fontsize=28, fontweight='bold')

# Show plot
plt.show()
