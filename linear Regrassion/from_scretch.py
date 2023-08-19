# import numpy as np
# import matplotlib.pyplot as plt
#
# # Generate some example data
# #The seed is like the starting point for those rules. If you use the same seed, you'll always get the same sequence of numbers from the machine.
#
# np.random.seed(0)
#
# #generates an array X containing 100 random numbers between 0 and 10.
#
# X = np.random.rand(100, 1) * 10  # Random features (e.g., house size)
# # X is independent variable example housing size,area anything
#
# #creates an array y where each value is obtained by applying a linear equation to the values in X, and then adding some random noise to the linear relationship.
#
# y = 2 * X + 3 + np.random.randn(100, 1) * 2  # True relationship with noise
# #y is price of the house
#
# # Initialize parameters
# m = 0.1  # Initial slope
# b = 0.1  # Initial intercept
# learning_rate = 0.01
# epochs = 1000   #This is the number of times the gradient descent algorithm iterates over the entire dataset. Each iteration is called an epoch
#
# # Gradient Descent
# for _ in range(epochs):
#     y_pred = m * X + b  # Predicted values of price calculated by program
#     error = y_pred - y   # Error between predicted and actual values
#
#     # Update parameters using gradients
#     m -= learning_rate * np.mean(error * X)
#     b -= learning_rate * np.mean(error)
#
# # Make predictions
# new_X = np.array([[4.5]])  # New house size for prediction
# predicted_price = m * new_X + b
# print("Predicted Price:", predicted_price[0][0])
#
# # Plotting the results
# plt.scatter(X, y, label="Actual data")
# plt.plot(X, m * X + b, color='red', label="Linear regression")
# plt.xlabel("House Size")
# plt.ylabel("Price")
# plt.title("Linear Regression for House Price Prediction")
# plt.legend()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load the California Housing dataset
california_housing = fetch_california_housing()

# (features) data: This is an attribute of the dataset that contains the feature data. In the California housing dataset, the feature data includes various attributes or columns that describe
# the characteristics of housing districts, such as average rooms, average bedrooms, population,
X = california_housing.data

#.target: This is an attribute of the dataset that contains the target variable data. In the context of regression problems, the
# target variable is the one you're trying to predict.
y = california_housing.target

# Normalize the features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

# Add a column of ones to X for the intercept term
X_normalized = np.hstack((np.ones((X_normalized.shape[0], 1)), X_normalized))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Initialize parameters
num_features = X_train.shape[1]
theta = np.zeros(num_features)
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    y_pred = np.dot(X_train, theta)  # Predicted values
    error = y_pred - y_train          # Error between predicted and actual values

    # Update parameters using gradients
    gradient = np.dot(X_train.T, error) / len(X_train)
    theta -= learning_rate * gradient

# Make predictions on the test set
y_pred_test = np.dot(X_test, theta)

# Calculate the mean squared error on the test set
mse = np.mean((y_pred_test - y_test)**2)
print("Mean squared error:", mse)

# Plotting the results
plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
plt.show()