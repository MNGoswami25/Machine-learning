import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
#X is a modified version of the feature matrix from the diabetes dataset, focusing on only one specific feature column for further analysis or modeling.
X = diabetes.data[:, np.newaxis, 2]  # Use a single feature for simplicity
#it's the dependent variable that you'll be trying to predict or model based on the features in X.
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create polynomial features using PolynomialFeatures()
# if degree=2, the original features x1, x2, ... will be transformed into 1, x1, x2, x1^2, x1*x2, x2^2, and so on.
degree = 3
poly_features = PolynomialFeatures(degree=degree)
#  poly_features.fit_transform(X_train) is an array (X_train_poly) containing the original features as well as the polynomial features created based on the specified degree.
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Fit a linear regression model to the polynomial features
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict using the model
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculate Mean Squared Error for train and test sets
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Mean Squared Error (Train): {mse_train:.2f}")
print(f"Mean Squared Error (Test): {mse_test:.2f}")

# Plot the results
plt.scatter(X_train, y_train, label='Training data')
plt.scatter(X_test, y_test, label='Testing data', color='r')
plt.plot(X_test, y_test_pred, color='k', label='Polynomial regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression')
plt.show()