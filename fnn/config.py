import numpy as np

# Input data
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])

# Training labels for input data
y = np.array([[0, 1, 1, 1, 1, 1, 1, 0]]).T

# Number of hidden layers in network
layers = 3

# Number of nodes in a hidden layer
nodes = 3

# Learning rate for gradient descent
learning_rate = 0.05

# Number of training iterations
iterations = 100000
