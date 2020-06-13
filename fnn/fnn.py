import numpy as np
from config import *

def sigmoid(x):
    ''' Pass a value through the sigmoid function '''
    return 1/(1+np.exp(-x))

def initialise_network(X, y, layers=1, nodes=2):
    ''' Initialise a network with random weight vectors '''
    network = []
    # Iterate across each hidden layer and the output layer
    for layer in range(layers+1):
        # Weights between input layer and first hidden layer
        if layer == 0:
            network.append(2*np.random.random((X.shape[1]+1, nodes))-1)
        # Weights between last hidden layer and output layer
        elif layer == layers:
            network.append(2*np.random.random((nodes+1, y.shape[1]))-1)
        # Weights between consecutive hidden layers
        else:
            network.append(2*np.random.random((nodes+1, nodes))-1)
    return network

def calculate_outputs(network, X):
    ''' Calculate the outputs of each layer in the network '''
    outputs = []
    for layer in range(len(network)+1):
        # Outputs after input layer
        if layer == 0:
            outputs.append(np.hstack((np.ones((X.shape[0], 1)), X)))
        # Outputs after output layer
        elif layer == len(network):
            outputs.append(np.dot(outputs[-1], network[-1]))
        # Outputs after hidden layers
        else:
            outputs.append(np.hstack((np.ones((X.shape[0], 1)), sigmoid(np.dot(outputs[-1], network[layer-1])))))
    return outputs

def calculate_loss(network, outputs, y):
    ''' Calculate loss of each layer in the network '''
    loss = []
    for output in reversed(range(len(outputs))):
        # Loss of output layer
        if output == len(outputs)-1:
            loss.append(y-outputs[output])
        # Loss of hidden layers
        else:
            loss.append(
                outputs[output][:, 1:]
                *(1-outputs[output][:, 1:])
                *np.dot(loss[-1], network[output].T[:, 1:])
            )
    return list(reversed(loss))

def calculate_partial_derivatives(outputs, loss):
    ''' Calculate partial derivatives of each layer in the network '''
    partial_derivatives = []
    for output in range(1, len(outputs)):
        partial_derivatives.append(loss[output][: , np.newaxis, :]*outputs[output-1][:, :, np.newaxis])
    return partial_derivatives

def calculate_gradients(partial_derivatives):
    ''' Calculate gradients of each layer in the network'''
    gradients = []
    for partial_derivative in partial_derivatives:
        gradients.append(np.average(partial_derivative, axis=0))
    return gradients

def update_network(network, gradients, learning_rate):
    ''' Update network weights using gradient descent '''
    network_updated = network.copy()
    for layer in range(len(network)):
        network_updated[layer] += learning_rate*gradients[layer]
    return network_updated

def main(X, y, layers, nodes, learning_rate, iterations):
    np.random.seed(1)
    network = initialise_network(X, y, layers, nodes)
    for _ in range(iterations):
        outputs = calculate_outputs(network, X)
        loss = calculate_loss(network, outputs, y)
        partial_derivatives = calculate_partial_derivatives(outputs, loss)
        gradients = calculate_gradients(partial_derivatives)
        network = update_network(network, gradients, learning_rate)
    print(np.round(outputs[-1], 0))

main(X, y, layers, nodes, learning_rate, iterations)
