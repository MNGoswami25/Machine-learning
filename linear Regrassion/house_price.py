import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r'D:\Machine-learning\Housing.csv')

# 'price' is the target variable that we want to predict

# Separate features (X) and target (y)
X = data[['area']]
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Sort the values for smooth plotting of the regression line
sorted_indexes = np.argsort(X_test.values.flatten())
X_sorted = X_test.values[sorted_indexes]
y_pred_sorted = y_pred[sorted_indexes]

# Calculate the mean squared error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

new_area = 8100

# Reshape the new_area value to match the input shape expected by the model
new_area = np.array(new_area).reshape(-1, 1)

# Make the prediction using the trained model
predicted_price = model.predict(new_area)

print("Predicted Price:", predicted_price)
# Plot the data points
plt.scatter(X_test, y_test, label="Test data", alpha=0.7)

# Plot the linear regression line
plt.plot(X_sorted, y_pred_sorted, color='red', label="Linear regression")

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression for House Price Prediction")
plt.legend()
plt.show()