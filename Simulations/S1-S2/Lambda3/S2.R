# Load required package
if (!requireNamespace("Iso", quietly = TRUE)) {
  install.packages("Iso")
}
library(Iso)

# Parameters
n <- 1600
Lambda_1 <- seq(0.8, 10, length.out = 100)
num_reps <- 100
set.seed(123)

# Storage for results
crps_results <- numeric(num_reps)
max_sq_diff_results <- numeric(num_reps)

# Monte Carlo simulation
for (rep in 1:num_reps) {
  # Generate data
  a <- (n - seq(1, n)) / n
  b <- a + 1 
  y <- runif(n, min = a, max = b)
  
  # Split into training and testing sets
  train_idx <- sample(1:n, size = floor(0.75 * n))
  test_idx <- setdiff(1:n, train_idx)
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  a_test <- a[test_idx]
  b_test <- b[test_idx]
  
  # Apply isotonic regression on training data for each t
  F_hat_train <- sapply(Lambda_1, function(t) {
    Iso::pava(as.numeric(y_train <= t), decreasing = FALSE)
  })
  
  # Predict for test data using closest training value for each y_test
  F_hat_test <- sapply(Lambda_1, function(t) {
    sapply(y_test, function(y_i) {
      # Find the closest y_j in the training set to y_i
      closest_idx <- which.min(abs(y_train - y_i))
      F_hat_train[closest_idx, which(Lambda_1 == t)]
    })
  })
  
  # Calculate true CDF for test set
  # Calculate true CDF for test set
  F_true_test <- sapply(Lambda_1, function(t) {
    ifelse(t < a_test, 0, 
           ifelse(t > b_test, 1, (t - a_test) / (b_test - a_test)))
  })
  
  
  # CRPS Calculation
  crps <- mean(rowMeans((F_hat_test - F_true_test)^2))
  
  # Maximum Squared Difference Calculation
  max_sq_diff <- max(colMeans((F_hat_test - F_true_test)^2))
  
  # Store results
  crps_results[rep] <- crps
  max_sq_diff_results[rep] <- max_sq_diff
}

# Save results to CSV files
output_dir <- "/Users/carlos/Desktop/Scenario1 and 2/Lambda1"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE) # Create the directory if it doesn't exist
}

write.csv(crps_results, file.path(output_dir, "S2_n_1600_Lambda_3_crps.csv"), row.names = FALSE)
write.csv(max_sq_diff_results, file.path(output_dir, "S2_n_1600_Lambda_3_max_sq.csv"), row.names = FALSE)

# Report results
mean_crps <- mean(crps_results)
mean_max_sq_diff <- mean(max_sq_diff_results)

cat("Mean CRPS:", mean_crps, "\n")
cat("Mean Maximum Squared Difference:", mean_max_sq_diff, "\n")

