from sklearn.linear_model import LinearRegression
import numpy as np

# Sample input data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the output for a new input
new_X = np.array([[6]])
predicted_y = model.predict(new_X)

print("Predicted output:", predicted_y)