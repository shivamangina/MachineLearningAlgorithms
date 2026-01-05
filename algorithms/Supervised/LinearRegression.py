# TODO: write a Linear Regrassion algoritm for a house price prediction
# from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_url = 'https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv'
lifesat = pd.read_csv(data_url)

print(lifesat.head())


# Sample input data
X = lifesat['GDP per capita (USD)'].values
y = lifesat['Life satisfaction'].values

print(X)
print(y)

# create a scatter plot
plt.scatter(X, y)
plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life satisfaction')
plt.title('Scatter plot of GDP per capita vs Life satisfaction')
plt.show()


# TODO: Create a linear regression model
# model = LinearRegression()
# TODO: gradient decent and batch greedient descent and Stochastic

# # # Fit the model to the data
# model.fit(X, y)

# # # Predict the output for a new input
# india_gdp = 791.22
# new_X = np.array([[india_gdp]])
# predicted_y = model.predict(new_X)

# print("Predicted output:", predicted_y)

# Locally weighted linear regression

class LinearRegression:
    def __init__():
        print("initialize parameteres")
        
    def forward_pass():
        print("forward pass")
        
    def backward_pass():
        print("backward pass")
        
    def compute_loss():
        print("compute Loss")
        
    def update_parameters():
        print(0)
    
    def train():
        print("train")


# Build this

