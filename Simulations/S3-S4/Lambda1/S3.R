# Load required packages
if (!requireNamespace("glmgen", quietly = TRUE)) {
  install.packages("glmgen")
}
if (!requireNamespace("splines", quietly = TRUE)) {
  install.packages("splines")
}
library(glmgen)
library(splines)

# Parameters
n <- 1600
Lambda_1 <- seq(-1, 0.4, length.out = 100) # Evaluation points
num_reps <- 100
k <- 3 # Linear trend filtering
set.seed(123)

# Storage for results
crps_trend_results <- numeric(num_reps)
max_sq_diff_trend_results <- numeric(num_reps)
crps_spline_results <- numeric(num_reps)
max_sq_diff_spline_results <- numeric(num_reps)

# Monte Carlo simulation
for (rep in 1:num_reps) {
  # Generate data
  mu <- 1 + 0.5 * sin(2 * pi * seq(1, n) / n)
  y <- rexp(n, rate = 1 / mu)
  
  # Split into training and testing sets
  train_idx <- sample(1:n, size = floor(0.75 * n))
  test_idx <- setdiff(1:n, train_idx)
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  mu_test <- mu[test_idx]
  
  # Apply Trend Filtering for each t in Lambda
  # Apply Trend Filtering for each t in Lambda
  # Apply Trend Filtering for each t in Lambda
  F_hat_test_trend <- sapply(Lambda_1, function(t) {
    # Compute w_t for training data
    w_t <- as.numeric(y_train <= t)
    # Fit trend filtering model
    trend_model <- glmgen::trendfilter(x = w_t, y = NULL, k = k,lambda = exp(12))
    # Predict using test indices
    predict(trend_model, x.new = test_idx, lambda = exp(12))
  })
  F_hat_test_spline <- sapply(Lambda_1, function(t) {
    w_t <- as.numeric(y_train <= t) # Compute indicator vector
    #spline_model <- smooth.spline( w_t, spar = 0.78) # Fit smoothing spline
    spline_model <- smooth.spline( w_t, spar = 0.9) # Fit smoothing spline
    predict(spline_model, x = test_idx)$y # Predict for test indices
  })
  # Calculate true CDF for test set
  F_true_test <- outer(mu_test, Lambda_1, function(mu, t) pexp(t, rate = 1 / mu))
  
  # CRPS Calculation for Trend Filtering
  crps_trend <- mean(rowMeans((F_hat_test_trend - F_true_test)^2))
  max_sq_diff_trend <- max(colMeans((F_hat_test_trend - F_true_test)^2))
  
  # CRPS Calculation for Smoothing Splines
  crps_spline <- mean(rowMeans((F_hat_test_spline - F_true_test)^2))
  max_sq_diff_spline <- max(colMeans((F_hat_test_spline - F_true_test)^2))
  
  # Store results
  crps_trend_results[rep] <- crps_trend
  max_sq_diff_trend_results[rep] <- max_sq_diff_trend
  crps_spline_results[rep] <- crps_spline
  max_sq_diff_spline_results[rep] <- max_sq_diff_spline
}
# Save results to CSV files
output_dir <- "/Users/carlos/Desktop/Scenario 3"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE) # Create the directory if it doesn't exist
}


# Save results to CSV files
write.csv(crps_trend_results, "S3_n_1600_Lambda_1_trend_crps.csv", row.names = FALSE)
write.csv(max_sq_diff_trend_results, "S3_n_1600_Lambda_1_trend_max_sq.csv", row.names = FALSE)
write.csv(crps_spline_results, "S3_n_1600_Lambda_1_spline_crps.csv", row.names = FALSE)
write.csv(max_sq_diff_spline_results, "S3_n_1600_Lambda_1_spline_max_sq.csv", row.names = FALSE)

# Report results
mean_crps_trend <- mean(crps_trend_results)
mean_max_sq_diff_trend <- mean(max_sq_diff_trend_results)
mean_crps_spline <- mean(crps_spline_results)
mean_max_sq_diff_spline <- mean(max_sq_diff_spline_results)

cat("Trend Filtering - Mean CRPS:", mean_crps_trend, "\n")
cat("Trend Filtering - Mean Maximum Squared Difference:", mean_max_sq_diff_trend, "\n")
cat("Smoothing Splines - Mean CRPS:", mean_crps_spline, "\n")
cat("Smoothing Splines - Mean Maximum Squared Difference:", mean_max_sq_diff_spline, "\n")
