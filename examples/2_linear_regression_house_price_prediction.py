from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Get the Data
data_url = 'https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv'
lifesat = pd.read_csv(data_url)

print(lifesat.head())


# Sample input data
X = lifesat['GDP per capita (USD)'].values.reshape(-1, 1)
y = lifesat['Life satisfaction'].values

print(X)
print(y)

# create a scatter plot
plt.scatter(X, y)
plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life satisfaction')
plt.title('Scatter plot of GDP per capita vs Life satisfaction')
plt.show()



# Create a linear regression model
model = LinearRegression()

# # Fit the model to the data
model.fit(X, y)

# # Predict the output for a new input
india_gdp = 791.22
new_X = np.array([[india_gdp]])
predicted_y = model.predict(new_X)

print("Predicted output:", predicted_y)
