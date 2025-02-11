# Load required packages
if (!requireNamespace("glmgen", quietly = TRUE)) install.packages("glmgen")
if (!requireNamespace("splines", quietly = TRUE)) install.packages("splines")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("gridExtra", quietly = TRUE)) install.packages("gridExtra")
if (!requireNamespace("viridis", quietly = TRUE)) install.packages("viridis")

library(glmgen)
library(splines)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(viridis)
library(maps)

# Load the dataset
data <- read.csv("daily_44201_2024.csv")

# Remove rows with missing latitude and longitude values
data_filtered <- data %>%
  filter(!is.na(Latitude) & !is.na(Longitude))

# Binning latitudes and longitudes in lexicographic order
data_filtered <- data_filtered %>%
  arrange(Latitude, Longitude)

# Create unique identifiers for each location (Latitude, Longitude)
data_filtered$location_id <- as.integer(as.factor(paste(data_filtered$Latitude, data_filtered$Longitude)))

# Aggregate data: Compute the mean of Arithmetic Mean for each (Latitude, Longitude)
agg_data <- data_filtered %>%
  group_by(location_id, Latitude, Longitude) %>%
  summarize(y_i = mean(Arithmetic.Mean, na.rm = TRUE), .groups = 'drop')

y <- agg_data$y_i
n <- length(y)

# Set evaluation points Lambda
Lambda_1 <- seq(0, 1, length.out = 100)

# Number of repetitions for Monte Carlo simulation
num_reps <- 1
set.seed(123)

# Storage for results
crps_trend_results <- numeric(num_reps)
max_sq_diff_trend_results <- numeric(num_reps)
crps_spline_results <- numeric(num_reps)
max_sq_diff_spline_results <- numeric(num_reps)

# Monte Carlo simulation
for (rep in 1:num_reps) {
  cat("Running iteration:", rep, "\n")
  
  # Split data into training and testing
  train_idx <- sample(1:n, size = floor(0.75 * n))
  test_idx <- setdiff(1:n, train_idx)
  
  y_train <- y[train_idx]
  y_test <- y[test_idx]
  
  # Apply Trend Filtering for each t in Lambda
  F_hat_test_trend <- sapply(Lambda_1, function(t) {
    w_t <- as.numeric(y_train <= t)
    trend_model <- glmgen::trendfilter(x = w_t, y = NULL, k = 3, lambda = exp(10))
    predict(trend_model, x.new = 1:length(y_test), lambda = exp(10))
  })
  
  # Apply Smoothing Splines for each t in Lambda
  F_hat_test_spline <- sapply(Lambda_1, function(t) {
    w_t <- as.numeric(y_train <= t)
    spline_model <- smooth.spline(w_t, spar = 0.05)
    predict(spline_model, x = 1:length(y_test))$y
  })
  
  # Empirical estimation of CDF for test set
  F_true_test <- sapply(Lambda_1, function(t) {
    rowMeans(outer(y_test, t, function(y, t_val) y <= t_val))
  })
  
  # Compute CRPS for Trend Filtering
  crps_trend <- mean(rowMeans((F_hat_test_trend - F_true_test)^2))
  max_sq_diff_trend <- max(colMeans((F_hat_test_trend - F_true_test)^2))
  
  # Compute CRPS for Smoothing Splines
  crps_spline <- mean(rowMeans((F_hat_test_spline - F_true_test)^2))
  max_sq_diff_spline <- max(colMeans((F_hat_test_spline - F_true_test)^2))
  
  # Store results
  crps_trend_results[rep] <- crps_trend
  max_sq_diff_trend_results[rep] <- max_sq_diff_trend
  crps_spline_results[rep] <- crps_spline
  max_sq_diff_spline_results[rep] <- max_sq_diff_spline
}

# Save results to CSV files
output_dir <- "/Users/carlos/Desktop/ReaData"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

#write.csv(crps_trend_results, file.path(output_dir, "real_data_ozone_trend_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_trend_results, file.path(output_dir, "real_data_ozone_trend_max_sq.csv"), row.names = FALSE)
#write.csv(crps_spline_results, file.path(output_dir, "real_data_ozone_spline_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_spline_results, file.path(output_dir, "real_data_ozone_spline_max_sq.csv"), row.names = FALSE)

# Report results
cat("Trend Filtering - Mean CRPS:", mean(crps_trend_results), "\n")
cat("Trend Filtering - Mean Maximum Squared Difference:", mean(max_sq_diff_trend_results), "\n")
cat("Smoothing Splines - Mean CRPS:", mean(crps_spline_results), "\n")
cat("Smoothing Splines - Mean Maximum Squared Difference:", mean(max_sq_diff_spline_results), "\n")
