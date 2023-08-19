import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
# make_classification is a function from scikit-learn that is used to generate a synthetic dataset for binary classification tasks.
#Overall, X represents the feature matrix, and y represents the target labels
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sigmoid function -- transforming linear combinations of features into probabilities that help make class predictions.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
#num_features is a variable that represents the number of features in the training dataset (X_train). It's calculated using the .shape attribute of the X_train array.
num_features = X_train.shape[1]
#represents the coefficients or weights used in logistic regression to make predictions based on the input features
theta = np.zeros((num_features, 1))
learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    #calculates the dot product of the feature matrix X_train and the parameter vector theta
    z = np.dot(X_train, theta)
    #sigmoid function to the linear combinations z, which converts them into predicted probabilities between 0 and 1.
    predictions = sigmoid(z)
    error = predictions - y_train.reshape(-1, 1)
    #  This computes the gradient of the cost function with respect to the parameter vector theta
    gradient = np.dot(X_train.T, error) / len(y_train)
    theta -= learning_rate * gradient

# Make predictions on the test set
test_predictions = sigmoid(np.dot(X_test, theta))
test_predictions = np.round(test_predictions)  # Convert probabilities to binary predictions (0 or 1)

# Calculate accuracy
accuracy = accuracy_score(y_test, test_predictions)
print("Accuracy:",accuracy)