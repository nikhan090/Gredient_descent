import numpy as np
import matplotlib.pyplot as plt

# Define the dataset
X = np.array([500, 1000, 1500, 2000])  # size in square feet
y = np.array([50, 100, 150, 200])      # price in $1000s

# Normalize the input features for better gradient descent performance
X_mean = np.mean(X)
print(X_mean)
X_std = np.std(X)
print(X_std)
X_norm = (X - X_mean) / X_std
# print(X_norm)

# Y_mean = np.mean(y)
# print(Y_mean)
# Y_std = np.std(y)
# print(Y_std)
# Y_norm = (y - Y_mean) / Y_std
# plt.plot(X_norm, Y_norm)
plt.plot(X_norm,y)
plt.show()
# Add bias term (intercept)
X_b = np.c_[np.ones(len(X_norm)), X_norm]  # shape: (m, 2)


# Initialize parameters (theta0 = intercept, theta1 = slope)
theta = np.zeros(2)

# Gradient Descent settings
alpha = 0.1  # learning rate
iterations = 20
m = len(y)  # number of training examples
cost_history = []

# Function to compute cost
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
    return cost

# Gradient Descent Algorithm
iteration_values = []

for i in range(iterations):
    predictions = X_b.dot(theta)
    errors = predictions - y
    gradients = (1 / m) * X_b.T.dot(errors)
    theta -= alpha * gradients
    cost = compute_cost(X_b, y, theta)
    cost_history.append(cost)
    iteration_values.append((i + 1, theta[0], theta[1], cost))

import pandas as pd
iteration_df = pd.DataFrame(iteration_values, columns=["Iteration", "Theta 0", "Theta 1", "Cost"])
print(iteration_df)


theta, cost_history[-1]  # final parameters and cost
