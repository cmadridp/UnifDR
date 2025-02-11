# Load required packages
if (!requireNamespace("rpart", quietly = TRUE)) {
  install.packages("rpart")
}
if (!requireNamespace("earth", quietly = TRUE)) {
  install.packages("earth")
}
if (!requireNamespace("randomForest", quietly = TRUE)) {
  install.packages("randomForest")
}
if (!requireNamespace("drf", quietly = TRUE)) {
  install.packages("drf")
}
if (!requireNamespace("engression", quietly = TRUE)) {
  install.packages("engression")
}

library(rpart)
library(earth)
library(randomForest)
library(drf)
library(engression)

# Parameters
set.seed(123)
n <- 400
Lambda_1 <- seq(0.8, 10, length.out = 100) # Evaluation points
num_reps <- 100
train_ratio <- 0.75

# Create output directory for results
output_dir <- "/Users/carlos/Desktop/Scenario 5/Lambda3"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Storage for results
crps_cart_results <- numeric(num_reps)
max_sq_diff_cart_results <- numeric(num_reps)
crps_mars_results <- numeric(num_reps)
max_sq_diff_mars_results <- numeric(num_reps)
crps_rf_results <- numeric(num_reps)
max_sq_diff_rf_results <- numeric(num_reps)
crps_drf_results <- numeric(num_reps)
max_sq_diff_drf_results <- numeric(num_reps)
crps_engression_results <- numeric(num_reps)
max_sq_diff_engression_results <- numeric(num_reps)

# Function to generate data for Scenario 5
generate_data <- function(n) {
  x <- matrix(runif(n * 5, 0, 1), nrow = n, ncol = 5)
  h_x <- -3 * x[, 1] + 2 * log(1 + x[, 2]) + x[, 3] + 5 * x[, 4] + x[, 5]^2
  y <- rnorm(n, mean = h_x, sd = 1)
  list(x = x, y = y, h_x = h_x)
}

# Monte Carlo simulation
for (rep in 1:num_reps) {
  cat("Running iteration:", rep, "\n")
  
  # Generate data
  data <- generate_data(n)
  x_data <- data$x
  y_data <- data$y
  h_x_true <- data$h_x
  
  # Split into training and testing sets
  train_idx <- sample(1:n, size = floor(train_ratio * n))
  test_idx <- setdiff(1:n, train_idx)
  
  x_train <- x_data[train_idx, ]
  y_train <- y_data[train_idx]
  x_test <- x_data[test_idx, ]
  y_test <- y_data[test_idx]
  mu_test <- h_x_true[test_idx]
  
  # Initialize matrices to store predictions
  F_hat_test_cart <- matrix(0, nrow = length(test_idx), ncol = length(Lambda_1))
  F_hat_test_mars <- matrix(0, nrow = length(test_idx), ncol = length(Lambda_1))
  F_hat_test_rf <- matrix(0, nrow = length(test_idx), ncol = length(Lambda_1))
  F_hat_test_drf <- matrix(0, nrow = length(test_idx), ncol = length(Lambda_1))
  F_hat_test_engression <- matrix(0, nrow = length(test_idx), ncol = length(Lambda_1))
  
  
  # Fit Distributional Random Forest model
  drf_model <- drf(X = x_train, Y = matrix(y_train, ncol = 1), num.trees = 1000, splitting.rule = "CART")
  drf_predictions <- predict(drf_model, newdata = x_test)
  weights <- as.matrix(drf_predictions$weights)
  
  # Fit Engression model
  engr_model <- engression(x_train, y_train, hidden_dim=64, num_layer = 4, num_epochs = 230,lr=1)
  
  # Generate samples from the estimated conditional distribution
  Yhat_samples <- predict(engr_model, x_test, type = "sample", nsample = 270)
  
  
  
  for (j in seq_along(Lambda_1)) {
    t <- Lambda_1[j]
    
    # Construct response vector w_t
    w_t <- as.numeric(y_train <= t)
    
    # Fit CART model
    cart_model <- rpart(w_t ~ ., data = data.frame(x_train, w_t), control = rpart.control(cp = 0.000001))
    F_hat_test_cart[, j] <- predict(cart_model, newdata = data.frame(x_test), type = "vector")
    
    # Fit MARS model
    mars_model <- earth(w_t ~ ., data = data.frame(x_train, w_t), penalty = 0.9)
    F_hat_test_mars[, j] <- predict(mars_model, newdata = data.frame(x_test))
    
    # Fit Random Forest model for regression
    rf_model <- randomForest(x_train, w_t, ntree = 500, nodesize = 5)
    
    # Predict continuous values
    F_hat_test_rf[, j] <- predict(rf_model, newdata = x_test)
    
    
    F_hat_test_drf[, j] <- weights %*% w_t  # Estimate conditional CDF
    
    
    
    # Estimate conditional CDF using empirical probability
    F_hat_test_engression[, j] <- rowMeans(Yhat_samples <= t)
  }
  
  # Calculate true CDF for test set using normal distribution
  F_true_test <- outer(mu_test, Lambda_1, function(mu, t) pnorm(t, mean = mu, sd = 1))
  
  # CRPS Calculation
  crps_cart <- mean(rowMeans((F_hat_test_cart - F_true_test)^2))
  crps_mars <- mean(rowMeans((F_hat_test_mars - F_true_test)^2))
  crps_rf <- mean(rowMeans((F_hat_test_rf - F_true_test)^2))
  crps_drf <- mean(rowMeans((F_hat_test_drf - F_true_test)^2))
  crps_engression <- mean(rowMeans((F_hat_test_engression - F_true_test)^2))
  
  # Maximum Squared Difference Calculation
  max_sq_diff_cart <- max(colMeans((F_hat_test_cart - F_true_test)^2))
  max_sq_diff_mars <- max(colMeans((F_hat_test_mars - F_true_test)^2))
  max_sq_diff_rf <- max(colMeans((F_hat_test_rf - F_true_test)^2))
  max_sq_diff_drf <- max(colMeans((F_hat_test_drf - F_true_test)^2))
  max_sq_diff_engression <- max(colMeans((F_hat_test_engression - F_true_test)^2))
  
  # Store results
  crps_cart_results[rep] <- crps_cart
  max_sq_diff_cart_results[rep] <- max_sq_diff_cart
  crps_mars_results[rep] <- crps_mars
  max_sq_diff_mars_results[rep] <- max_sq_diff_mars
  crps_rf_results[rep] <- crps_rf
  max_sq_diff_rf_results[rep] <- max_sq_diff_rf
  crps_drf_results[rep] <- crps_drf
  max_sq_diff_drf_results[rep] <- max_sq_diff_drf
  crps_engression_results[rep] <- crps_engression
  max_sq_diff_engression_results[rep] <- max_sq_diff_engression
}




# Save results to CSV files
#write.csv(crps_cart_results, file.path(output_dir, "S5_n_1600_Lambda_3_cart_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_cart_results, file.path(output_dir, "S5_n_1600_Lambda_3_cart_max_sq.csv"), row.names = FALSE)
#write.csv(crps_mars_results, file.path(output_dir, "S5_n_1600_Lambda_3_mars_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_mars_results, file.path(output_dir, "S5_n_400_Lambda_3_mars_max_sq.csv"), row.names = FALSE)
#write.csv(crps_rf_results, file.path(output_dir, "S5_n_1600_Lambda_3_rf_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_rf_results, file.path(output_dir, "S5_n_1600_Lambda_3_rf_max_sq.csv"), row.names = FALSE)
#write.csv(crps_drf_results, file.path(output_dir, "S5_n_1600_Lambda_3_drf_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_drf_results, file.path(output_dir, "S5_n_1600_Lambda_3_drf_max_sq.csv"), row.names = FALSE)
#write.csv(crps_engression_results, file.path(output_dir, "S5_n_1600_Lambda_3_engression_crps.csv"), row.names = FALSE)
#write.csv(max_sq_diff_engression_results, file.path(output_dir, "S5_n_1600_Lambda_3_engression_max_sq.csv"), row.names = FALSE)


# Report mean results
cat("CART - Mean CRPS:", mean(crps_cart_results), "\n")
cat("CART - Mean Maximum Squared Difference:", mean(max_sq_diff_cart_results), "\n")
cat("MARS - Mean CRPS:", mean(crps_mars_results), "\n")
cat("MARS - Mean Maximum Squared Difference:", mean(max_sq_diff_mars_results), "\n")
cat("Random Forest - Mean CRPS:", mean(crps_rf_results), "\n")
cat("Random Forest - Mean Maximum Squared Difference:", mean(max_sq_diff_rf_results), "\n")
cat("DRF - Mean CRPS:", mean(crps_drf_results), "\n")
cat("DRF - Mean Maximum Squared Difference:", mean(max_sq_diff_drf_results), "\n")
cat("Engression - Mean CRPS:", mean(crps_engression_results), "\n")
cat("Engression - Mean Maximum Squared Difference:", mean(max_sq_diff_engression_results), "\n")
