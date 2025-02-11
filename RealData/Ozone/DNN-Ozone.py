import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
file_path = "daily_44201_2024.csv"
data = pd.read_csv(file_path)

# Preprocess the data: Remove rows with missing latitude, longitude, or ozone measurements
data = data.dropna(subset=['Latitude', 'Longitude', 'Arithmetic Mean'])

# Create a unique station identifier
data['location_id'] = data.groupby(['Latitude', 'Longitude']).ngroup()

# Aggregate data: Compute mean ozone level for each station
agg_data = data.groupby(['location_id', 'Latitude', 'Longitude']).agg(
    y_i=('Arithmetic Mean', 'mean'),  # Response: Mean ozone level
    mean_AQI=('AQI', 'mean'),
    mean_Observation_Percent=('Observation Percent', 'mean'),
    mean_1st_Max_Value=('1st Max Value', 'mean'),
    mean_1st_Max_Hour=('1st Max Hour', 'mean'),
    mean_Observation_Count=('Observation Count', 'mean')
).reset_index()

# Extract response and features
y = agg_data['y_i'].values
x = agg_data[['Latitude', 'Longitude', 'mean_AQI', 'mean_Observation_Percent', 
              'mean_1st_Max_Value', 'mean_1st_Max_Hour', 'mean_Observation_Count']].values


#print(np.max(y))
#print(np.min(y))
# Ensure data consistency and shape
n = len(y)
input_dim = x.shape[1]

# Set evaluation points Lambda
Lambda_1 = np.linspace(0, 1, 100)  # Adjust range based on ozone values

# Number of repetitions for Monte Carlo simulation
num_reps = 1
torch.manual_seed(1)
np.random.seed(1)

# Neural network hyperparameters
hidden_dim = 100
output_dim = 1
learning_rate = 0.001
num_epochs = 2000
num_layers = 2 # Number of hidden layers

# Storage for results
crps_dnn_results = np.zeros(num_reps)
max_sq_diff_dnn_results = np.zeros(num_reps)

# Define the neural network model
class OzoneNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(OzoneNN, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]  # First hidden layer

        for _ in range(num_layers - 1):  # Additional hidden layers
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid activation for CDF estimation

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
        model = OzoneNN(input_dim, hidden_dim, num_layers, output_dim).to(device)
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

#pd.DataFrame(crps_dnn_results, columns=["CRPS"]).to_csv(os.path.join(output_dir, "ozone_dnn_crps.csv"), index=False)
#pd.DataFrame(max_sq_diff_dnn_results, columns=["Max_Squared_Difference"]).to_csv(os.path.join(output_dir, "ozone_dnn_max_sq.csv"), index=False)

print("Results saved successfully.")

# Report results
print(f"DNN - Mean CRPS: {np.mean(crps_dnn_results):.4f}")
print(f"DNN - Mean Maximum Squared Difference: {np.mean(max_sq_diff_dnn_results):.4f}")
print(max_sq_diff_dnn_results)
