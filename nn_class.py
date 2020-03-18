import numpy as np
from random import random, seed


class NN:
    def __init__(self, dataset, n_hidden=0):
        self.dataset = dataset
        self.n_inputs = len(self.dataset[0]) - 1
        self.n_outputs = len(set([row[-1] for row in self.dataset]))

        self.network = list()
        self.hidden_layer = [{"weights": [random() for i in range(self.n_inputs + 1)]} for i in range(n_hidden)]
        self.network.append(self.hidden_layer)

        self.output_layer = [{"weights": [random() for i in range(n_hidden + 1)]} for i in range(self.n_outputs)]
        self.network.append(self.output_layer)

        self.network = np.squeeze(self.network)

    def activation(self, weights, inputs):
        self.activate = weights[-1]
        for i in range(len(weights) - 1):
            self.activate += weights[i] * inputs[i]

    def forward_prop(self, rows, activation):
        if activation == 'sigmoid':
            self.transfer = self.sigmoid
        if activation == 'relu':
            self.transfer = self.relu
        inputs = rows
        for layers in self.network:
            new_input = []
            for neuron in layers:
                self.activation(neuron['weights'], inputs)
                neuron['output'] = self.transfer(self.activate)
                new_input.append(neuron['output'])
            inputs = new_input
        return inputs

    def back_prop(self, expected, activation):
        if activation == 'sigmoid':
            self.derivative = self.sigmoid_deriv
        if activation == 'relu':
            self.derivative = self.relu_deriv
        for i in reversed(range(len(self.network))):
            layers = self.network[i]
            errors = list()

            if i != len(self.network) - 1:
                for j in range(len(layers)):
                    error = 0
                    for neuron in self.network[i + 1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layers)):
                    neuron = layers[j]
                    errors.append(expected[j] - neuron['output'])
            for j in range(len(layers)):
                neuron = layers[j]
                neuron['delta'] = errors[j] * self.derivative(neuron['output'])

    def update_weights(self, rows, lr):
        for i in range(len(self.network)):
            inputs = rows[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i - 1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += lr * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += lr * neuron['delta']

    def train_network(self, lr, epoch, activation='sigmoid'):
        for epochs in range(epoch):
            sum_error = 0
            for rows in self.dataset:
                outputs = self.forward_prop(rows, activation)  # output from forward prop.
                expected_output = [0 for i in range(self.n_outputs)]  # create list of zeros
                expected_output[rows[-1]] = 1  # This is because of how our train dataset looks like
                sum_error += sum([(expected_output[i] - outputs[i]) ** 2 for i in range(len(expected_output))])
                self.back_prop(expected_output, activation)  # back prop.
                self.update_weights(rows, lr)  # update based on back prop.
            print('>Epoch=%d, learning_rate=%.3f, error=%.3f' % (epochs + 1, lr, sum_error))

    def predict(self, test):
        outputs = self.forward_prop(test, self.activation)
        outputs = list(self.softmax(outputs))
        return outputs, outputs.index(max(outputs))

    def show_network(self):
        for layer in self.network:
            print(layer)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_deriv(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        if x > 0:
            return x
        else:
            return 0

    @staticmethod
    def relu_deriv(x):
        if x > 0:
            return 1
        else:
            return 0

    @staticmethod
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))


dataset = [[2.8, 2.6, 0],
           [1.5, 2.4, 0],
           [3.4, 4.4, 0],
           [1.4, 1.8, 0],
           [3.1, 3.0, 0],
           [7.6, 2.8, 1],
           [5.3, 2.1, 1],
           [6.9, 1.8, 1],
           [8.7, 0.2, 1],
           [7.7, 3.5, 1]]

my_nn = NN(dataset, n_hidden=3)
my_nn.train_network(lr=0.01, epoch=250, activation='sigmoid')
pred_bin, pred = my_nn.predict([1.4, 1.8])
print(pred_bin, pred)
