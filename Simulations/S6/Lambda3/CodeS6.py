import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from scipy.stats import chi2

# Parameters
n = 1600
Lambda_1 = np.linspace(0.8, 10, 100)  # Evaluation points
num_reps = 100
input_dim = 10
hidden_dim = 64
output_dim = 1
learning_rate = 0.001
num_epochs = 75#70#92#80
set_seed = 123

# Set random seed for reproducibility
torch.manual_seed(set_seed)
np.random.seed(set_seed)

# Use GPU if available for acceleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Storage for results
crps_dnn_results = np.zeros(num_reps)
max_sq_diff_dnn_results = np.zeros(num_reps)

# Define the neural network model
class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Sigmoid activation for CDF estimation
        )

    def forward(self, x):
        return self.model(x)


# Function to generate data for the new scenario with Poisson distribution
import numpy as np
from scipy.stats import chi2

# Function to generate data for the new scenario with Chi-square distribution
def generate_data(n):
    x = np.random.uniform(0, 1, (n, 10))  # 10-dimensional covariates

    # Compute h(x) using a structured approach to ensure positivity
    h_x = np.log(
        np.abs(
            -0.5 * np.sum(np.sin(np.pi * x[:, :3]), axis=1) +  # Sine transformation on first 3 dimensions
            #0.5 * np.sum(np.tan(2 * np.pi * x[:, 3:9]), axis=1) +  # Tangent transformation on next 6 dimensions
            0.02 * np.sum(x[:, 3:9]) -
            0.5 *np.cos( x[:, 9])  # Linear term for the last dimension
        ) + 2  # Adding 2 to ensure degrees of freedom > 0
    )

    # Generate Chi-square distributed samples with degrees of freedom parameter h_x
    y = chi2.rvs(df=h_x, size=n)

    return x, y, h_x  # Return covariates, responses, and h(x)

# Monte Carlo simulation
for rep in range(num_reps):
    print(f"Iteration {rep+1}/{num_reps}")

    # Generate data
    x_data, y_data, h_x_true = generate_data(n)
    
    # Split into training and testing sets
    train_size = int(0.75 * n)
    train_idx = np.random.choice(n, train_size, replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    x_train, y_train = x_data[train_idx], y_data[train_idx]
    x_test, y_test = x_data[test_idx], y_data[test_idx]
    mu_test = h_x_true[test_idx]  # h_x_true is used for CDF calculation
    print(y_test)   
    # Convert data to PyTorch tensors and move to device
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

    # Store estimated CDFs for all Lambda values
    F_hat_test = np.zeros((len(test_idx), len(Lambda_1)))

    # Loop over all threshold values in parallel using batching
    for j, t in enumerate(Lambda_1):
        # Construct response vector w_t for current t
        w_t = torch.tensor((y_train <= t).astype(float), dtype=torch.float32).to(device)

        # Train the neural network for the current t
        model = NN(input_dim, hidden_dim, output_dim).to(device)
        criterion = nn.BCELoss()  # Binary Cross Entropy for 0/1 responses
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop with gradient accumulation to speed up
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(x_train_tensor).squeeze()
            loss = criterion(outputs, w_t)
            loss.backward()
            optimizer.step()

        # Predict for test set and store results
        model.eval()
        with torch.no_grad():
            F_hat_test[:, j] = model(x_test_tensor).cpu().numpy().squeeze()

    # Calculate true CDF for test set using the Chi-square distribution
    F_true_test = np.array([
    [chi2.cdf(t, df=mu) for t in Lambda_1] for mu in mu_test
])
    # CRPS Calculation (optimized using NumPy broadcasting)
    crps_dnn = np.mean(np.mean((F_hat_test - F_true_test) ** 2, axis=1))

    # Maximum Squared Difference Calculation
    max_sq_diff_dnn = np.max(np.mean((F_hat_test - F_true_test) ** 2, axis=0))

    # Store results
    crps_dnn_results[rep] = crps_dnn
    max_sq_diff_dnn_results[rep] = max_sq_diff_dnn

# Save results to CSV files
output_dir = "/Users/carlos/Desktop/Scenario 6/Lambda3"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save results to CSV files
pd.DataFrame(crps_dnn_results, columns=["CRPS"]).to_csv(os.path.join(output_dir, "S6_n_1600_Lambda_3_dnn_crps.csv"), index=False)
pd.DataFrame(max_sq_diff_dnn_results, columns=["Max_Squared_Difference"]).to_csv(os.path.join(output_dir, "S6_n_1600_Lambda_3_dnn_max_sq.csv"), index=False)

print("Results saved successfully.")

# Report results
mean_crps_dnn = np.mean(crps_dnn_results)
mean_max_sq_diff_dnn = np.mean(max_sq_diff_dnn_results)

print(f"DNN - Mean CRPS: {mean_crps_dnn:.8f}")
print(f"DNN - Mean Maximum Squared Difference: {mean_max_sq_diff_dnn:.8f}")
print(crps_dnn_results)