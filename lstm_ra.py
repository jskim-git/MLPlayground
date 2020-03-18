import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

"""
Creating an LSTM model from scratch
Does that make sense or what... Seems very difficult!
"""

input_units = 2
hidden_units = 10
output_units = 5  # number of classifications
learning_rate = 0.001
beta1 = 0.90  # parameter V for Adam Optimizer
beta2 = 0.99  # parameter S for Adam Optimizer


# Defining Activation functions and their derivatives (for backpropagation)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    expX = np.exp(x)
    expX_sum = np.sum(expX, axis=1).reshape(-1, 1)
    expX = expX / expX_sum
    return expX


# Activation functions, showing activation functions
# Sigmoid(x)
x = np.linspace(-15, 15, 31)
y = []
for X in x:
    y.append(sigmoid(X))

plt.plot(x, y)
plt.show()

# tanh(x)
y = []
for X in x:
    y.append(tanh(X))

plt.plot(x, y)
plt.show()

