import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# Load the dataset
data = pd.read_csv(r'D:\Machine-learning\Housing.csv')

# Assume the dataset has columns like 'area', 'bedrooms', 'bathrooms', 'age', etc.
# 'price' is the target variable that we want to predict

# Separate features (X) and target (y)
X = data[['area', 'bedrooms', 'bathrooms', 'stories','mainroad','guestroom', 'basement', 'hotwaterheating','parking','prefarea','furnishingstatus']]
y = data['price']
# Convert categorical variables to one-hot encoded columns
X_encoded = pd.get_dummies(X, columns=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'parking', 'prefarea', 'furnishingstatus'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error to evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
# Plotting the actual vs. predicted prices
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label="Linear Regression Line")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend()
plt.show()