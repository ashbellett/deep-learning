""" Python implementation of a feed-forward neural network. """

import numpy as np

from numpy.random import random
from tqdm import tqdm
from config import *


def get_data(file_name):
    """ Get matrix of input data from CSV file """
    data = np.genfromtxt(file_name, dtype=float, delimiter=",")
    X = data[:, 0:-1]
    y = data[:, -1, np.newaxis]
    return (X, y)


def split_data(X, y, test_size):
    """ Split a proportion of the training set from its labels """
    data = np.hstack((X, y))
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1, np.newaxis]
    X_test = X[0:test_size, :]
    y_test = y[0:test_size, :]
    X_train = X[test_size:, :]
    y_train = y[test_size:, :]
    return (X_train, y_train, X_test, y_test)


def create_batches(X, y, batch_size=10):
    """ Create a batch of samples and labels """
    data = np.hstack((X, y))
    np.random.shuffle(data)
    batch_count = int(data.shape[0] / batch_size)
    batches = []
    for count in range(batch_count):
        batch = data[count * batch_size : (count + 1) * batch_size, :]
        X_batch = batch[:, :-1]
        y_batch = batch[:, -1, np.newaxis]
        batches.append((X_batch, y_batch))
    return batches


def sigmoid(x):
    """ Pass a value through the sigmoid function """
    return 1 / (1 + np.exp(-x))


def initialise_network(X, y, layers=(2,)):
    """ Initialise a network with random weight vectors """
    # Weights between input layer and first hidden layer
    network = [2 * random((X.shape[1] + 1, layers[0])) - 1]
    # Iterate across each hidden layer
    for layer, nodes in enumerate(layers):
        if layer == len(layers) - 1:
            # Weights between last hidden layer and output layer
            network.append(2 * random((nodes + 1, y.shape[1])) - 1)
        else:
            # Weights between consecutive hidden layers
            network.append(2 * random((nodes + 1, layers[layer + 1])) - 1)
    return network


def get_outputs(network, X):
    """ Calculate the outputs of each layer in the network """
    outputs = []
    for layer in range(len(network) + 1):
        # Outputs after input layer
        if layer == 0:
            outputs.append(np.hstack((np.ones((X.shape[0], 1)), X)))
        # Outputs after output layer
        elif layer == len(network):
            outputs.append(np.dot(outputs[-1], network[-1]))
        # Outputs after hidden layers
        else:
            outputs.append(
                np.hstack(
                    (
                        np.ones((X.shape[0], 1)),
                        sigmoid(np.dot(outputs[-1], network[layer - 1])),
                    )
                )
            )
    return outputs


def get_losses(network, outputs, y):
    """ Calculate losses of each layer in the network """
    losses = []
    for output in reversed(range(len(outputs))):
        # Losses of output layer
        if output == len(outputs) - 1:
            losses.append(y - outputs[output])
        # Losses of hidden layers
        else:
            losses.append(
                outputs[output][:, 1:]
                * (1 - outputs[output][:, 1:])
                * np.dot(losses[-1], network[output].T[:, 1:])
            )
    return list(reversed(losses))


def get_partial_derivatives(outputs, losses):
    """ Calculate partial derivatives of each layer in the network """
    partial_derivatives = []
    for output in range(1, len(outputs)):
        partial_derivatives.append(
            losses[output][:, np.newaxis, :] * outputs[output - 1][:, :, np.newaxis]
        )
    return partial_derivatives


def get_gradients(partial_derivatives):
    """ Calculate gradients of each layer in the network"""
    gradients = []
    for partial_derivative in partial_derivatives:
        gradients.append(np.average(partial_derivative, axis=0))
    return gradients


def update_network(network, gradients, learning_rate):
    """ Update network weights using gradient descent """
    network_updated = network.copy()
    for layer in range(len(network)):
        network_updated[layer] += learning_rate * gradients[layer]
    return network_updated


def get_predictions(network, x):
    """ Pass unlabelled data to get predicted labels """
    return get_outputs(network, x)[-1]


def main(file_name, test_size, batch_size, layers, learning_rate, iterations):
    """ Orchestrate forward and backward passes of the network """
    np.random.seed(1)
    X, y = get_data(file_name)
    X_train, y_train, X_test, y_test = split_data(X, y, test_size)
    network = initialise_network(X_train, y_train, layers)

    for _ in tqdm(range(iterations)):
        batches = create_batches(X_train, y_train, batch_size)
        for batch in batches:
            X, y = batch
            outputs = get_outputs(network, X)
            losses = get_losses(network, outputs, y)
            partial_derivatives = get_partial_derivatives(outputs, losses)
            gradients = get_gradients(partial_derivatives)
            network = update_network(network, gradients, learning_rate)

    test_error = []
    test_correct = 0
    for index, x in enumerate(X_test):
        prediction = get_predictions(network, np.array([x]))
        test_error.append(np.around(prediction) - y_test[index])
        if np.around(prediction) - y_test[index] == 0:
            test_correct += 1
    print("Test loss: ", np.average(test_error))
    print("Test accuracy: ", str(100 * test_correct / test_size) + "%")


if __name__ == "__main__":
    main(file_name, test_size, batch_size, layers, learning_rate, iterations)
