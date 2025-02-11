import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = pd.read_csv("chicago2015.csv", dtype={'Arrest': str})

# Preprocess the data: Remove rows with missing values in selected columns
data.dropna(subset=['Latitude', 'Longitude', 'Date', 'Beat', 'Arrest'], inplace=True)

# Convert 'Date' column to datetime and extract day of the week (to match R's Sunday = 1, Monday = 2, ..., Saturday = 7)
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %I:%M:%S %p')
data['DayOfWeek'] = data['Date'].dt.weekday + 2  # Shift by +2 to align with R (Monday=2, ..., Sunday=1)
data['DayOfWeek'] = data['DayOfWeek'].apply(lambda x: x if x <= 7 else 1)  # Adjust for Sunday (Python's 6 -> R's 1)

# Convert 'Arrest' column to numeric (True -> 1, False -> 0)
data['Arrest'] = data['Arrest'].apply(lambda x: 1 if x.strip().upper() == 'TRUE' else 0)

# Binning latitudes and longitudes into a 100x100 grid
data['lat_bin'] = pd.cut(data['Latitude'], bins=100, labels=False)
data['lon_bin'] = pd.cut(data['Longitude'], bins=100, labels=False)

# Create grid index and count crimes per cell
data['grid_cell'] = data['lat_bin'].astype(str) + "_" + data['lon_bin'].astype(str)
crime_counts = data.groupby('grid_cell').size().reset_index(name='count')

# Take the log of crime counts to get the response variable y_i
crime_counts['log_count'] = np.log(crime_counts['count'] + 1)

# Merge back to get covariates
data = data.drop_duplicates(subset=['grid_cell']).merge(crime_counts, on='grid_cell')

# Extract response and features
y = data['log_count'].values
x = data[['lon_bin', 'lat_bin', 'DayOfWeek', 'Beat', 'Arrest']].values
#print(x[1:10])
# Ensure data consistency and shape
n = len(y)
input_dim = x.shape[1]
#print(y)

# Set evaluation points Lambda
Lambda_1 = np.linspace(-1, 6, 100)
#print(min(y))
#print(max(y))

# Number of repetitions for Monte Carlo simulation
num_reps = 100
torch.manual_seed(1)
np.random.seed(1)

# Neural network hyperparameters
hidden_dim =64
output_dim = 1
learning_rate = 0.001
num_epochs = 1000
num_layers = 5  # Add number of hidden layers here


# Storage for results
crps_dnn_results = np.zeros(num_reps)
max_sq_diff_dnn_results = np.zeros(num_reps)

# Define the neural network model with a customizable number of hidden layers
class CrimeNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CrimeNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]  # First hidden layer

        for _ in range(num_layers - 1):  # Add more hidden layers if needed
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Keep Sigmoid for CDF estimation

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Monte Carlo simulation
for rep in range(num_reps):
    print(f"Iteration {rep+1}/{num_reps}")

    # Split data into training and testing sets
    train_size = int(0.75 * n)
    train_idx = np.random.choice(n, train_size, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    # Store estimated CDFs for all Lambda values
    F_hat_test = np.zeros((len(test_idx), len(Lambda_1)))

    # Loop over all threshold values in parallel using batching
    for j, t in enumerate(Lambda_1):
        # Construct response vector w_t for current threshold
        w_t = torch.tensor((y_train <= t).astype(float), dtype=torch.float32).to(device)

        # Train the neural network for the current threshold t
        model = CrimeNN(input_dim, hidden_dim, num_layers, output_dim).to(device)  # Now includes num_layers
        criterion = nn.BCELoss()  # Binary Cross Entropy for 0/1 responses
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(x_train_tensor).squeeze()
            loss = criterion(outputs, w_t)
            loss.backward()
            optimizer.step()

        # Predict for test set
        model.eval()
        with torch.no_grad():
            F_hat_test[:, j] = model(x_test_tensor).cpu().numpy().squeeze()

    # Compute empirical CDF for test set
    F_true_test = (y_test[:, None] <= Lambda_1).astype(int)

    # Compute CRPS
    crps_dnn = np.mean(np.mean((F_hat_test - F_true_test) ** 2, axis=1))

    # Compute Maximum Squared Difference
    max_sq_diff_dnn = np.max(np.mean((F_hat_test - F_true_test) ** 2, axis=0))

    # Store results
    crps_dnn_results[rep] = crps_dnn
    max_sq_diff_dnn_results[rep] = max_sq_diff_dnn


# Save results to CSV files
#output_dir = "/Users/carlos/Desktop/RealData"
#if not os.path.exists(output_dir):
#    os.makedirs(output_dir)

#pd.DataFrame(crps_dnn_results, columns=["CRPS"]).to_csv(os.path.join(output_dir, "chicago_dnn_crps.csv"), index=False)
#pd.DataFrame(max_sq_diff_dnn_results, columns=["Max_Squared_Difference"]).to_csv(os.path.join(output_dir, "chicago_dnn_max_sq.csv"), index=False)

print("Results saved successfully.")

# Report results
print(f"DNN - Mean CRPS: {np.mean(crps_dnn_results):.4f}")
print(f"DNN - Mean Maximum Squared Difference: {np.mean(max_sq_diff_dnn_results):.4f}")
print(max_sq_diff_dnn_results)
#print(F_true_test)


