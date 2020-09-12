""" Python implementation of a convolutional neural network. """

import math
import numpy as np
from config import *


def softmax(data):
    """ Pass data through softmax activation function """
    return np.exp(data) / np.sum(np.exp(data), axis=0)


def cross_entropy(probability):
    """ Calculate cross-entropy loss of probability value """
    return -np.log(probability)


def get_data(file_name):
    """ Get features and labels from a CSV file """
    data = np.genfromtxt(file_name, dtype=int, skip_header=1, delimiter=",")
    features = data[:, 1:]
    labels = data[:, 0]
    return (features, labels)


def scale_features(features):
    ''' Scale image pixel values about zero '''
    return np.array([image / 255 - 0.5 for image in features])


def reshape_features(features, image_shape):
    """ Reshape 1D feature vectors into 3D images """
    return np.array([image.reshape(image_shape) for image in features])


def prepare_features(features, image_shape):
    features = features.copy()
    features = scale_features(features)
    features = reshape_features(features, image_shape)
    return features


def initialise_filters(input_depth, filter_shape=(8, 3, 3)):
    """ Initialise convolutional filter weights using Gaussian distribution """
    filter_depth, filter_height, filter_width = filter_shape
    weights = np.random.randn(filter_depth, input_depth, filter_height, filter_width) / (input_depth * filter_height * filter_width)
    return weights


def initialise_weights(input_nodes, output_nodes):
    """ Initialise dense network weights using Gaussian distribution """
    weights = np.random.randn(input_nodes, output_nodes) / input_nodes
    biases = np.zeros(output_nodes)
    return (weights, biases)


def create_network(convolution_shape, convolution_stride, pooling_size, input_shape, output_shape):
    """ Apply network configuration to initialise network weights """
    image_depth, image_height, image_width = input_shape
    filter_depth, filter_height, filter_width = convolution_shape
    input_nodes = filter_depth * ((image_height - filter_height) // convolution_stride + 1) * ((image_width - filter_width) // convolution_stride + 1) // (pooling_size**2)
    network = [
        initialise_filters(image_depth, convolution_shape), # convolutional layer (filter weights)
        initialise_weights(input_nodes, output_shape) # dense layer (node weights and biases)
    ]
    return network


def forward_pass(network, image):
    """ Perform forward pass of network (calculate predictions) """
    convolved_image = convolution_forward(network[0], image)
    pooled_image = maxpooling_forward(convolved_image, pooling_size)
    outputs, activations = dense_forward(network[1], pooled_image)
    return outputs, activations, pooled_image, convolved_image


def backward_pass(network, gradient_loss_outputs, activations, pooled_image, convolved_image, image, pooling_size, learning_rate):
    """ Perform backward pass of network (calculate loss gradients and update network parameters) """
    network = network.copy()
    gradient_loss_dense, gradient_loss_activations = dense_backward(network[1], gradient_loss_outputs, pooled_image, activations)
    network[1] = dense_update(network[1], gradient_loss_activations, pooled_image, learning_rate)
    gradient_loss_maxpooling = maxpooling_backward(gradient_loss_dense, convolved_image, pooling_size)
    gradient_loss_filters = convolution_backward(network[0], gradient_loss_maxpooling, image)
    network[0] = convolution_update(network[0], gradient_loss_filters, learning_rate)
    return network


def train_network(network, features, labels, image_shape, pooling_size, learning_rate):
    """ Orchestrate forward and backward passes of network during training """
    network = network.copy()
    losses = []
    true_positives = 0
    for iteration, (image, label) in enumerate(zip(features, labels)):
        outputs, activations, pooled_image, convolved_image = forward_pass(network, image)
        gradient_loss_outputs = np.zeros(10)
        gradient_loss_outputs[label] = -1 / outputs[label]
        network = backward_pass(
            network,
            gradient_loss_outputs,
            activations, pooled_image,
            convolved_image,
            image,
            pooling_size,
            learning_rate
        )
        # Cross entropy loss function using predicted likelihood of correct class
        losses.append(cross_entropy(outputs[label]))
        true_positives += 1 if np.argmax(outputs) == label else 0
        if (iteration+1)%60 == 0:
            print('\033c')
            print('Training completion:\t', round(100*iteration/len(labels), 3), '%')
            print('Mean training loss:\t', round(sum(losses)/(iteration+1), 6))
            print('Training accuracy:\t', round(100*true_positives/(iteration+1), 3), '%')
    return network, outputs, losses, true_positives


def test_network(network, features, labels):
    """ Orchestrate forward pass of network during testing """
    losses = []
    true_positives = 0
    for iteration, (image, label) in enumerate(zip(features, labels)):
        outputs, _, _, _ = forward_pass(network, image)
        # Cross entropy loss function using predicted likelihood of correct class
        losses.append(cross_entropy(outputs[label]))
        true_positives += 1 if np.argmax(outputs) == label else 0
        if (iteration+1)%10 == 0:
            print('\033c')
            print('Testing completion:\t', round(100*iteration/len(labels), 3), '%')
            print('Mean testing loss:\t', round(sum(losses)/(iteration+1), 6))
            print('Testing accuracy:\t', round(100*true_positives/(iteration+1), 3), '%')
    return outputs, losses, true_positives


def convolution_forward(parameters, image_volume):
    """ Convolve a 3D image with a sequence of filters """
    image_depth, image_height, image_width = image_volume.shape
    filter_count, filter_depth, filter_height, filter_width = parameters.shape
    image_region = np.zeros((image_depth, filter_height, filter_width))
    output = np.zeros((filter_count, image_height - filter_height + 1, image_width - filter_width + 1))
    for filter_index, filter_volume in enumerate(parameters):
        for image_row in range(image_height - filter_height + 1):
            for image_column in range(image_width - filter_width + 1):
                # Get 3D image region volume
                image_region = image_volume[:image_depth, image_row : image_row + filter_height, image_column : image_column + filter_width]
                # Reshape image region from 3D to 1D
                image_vector = image_region.reshape(image_region.size)
                # Reshape filter from 3D to 1D
                filter_vector = filter_volume.reshape((filter_volume.size, 1))
                # Dot product of reshaped vectors is equivalent to convolution of volumes
                output[filter_index, image_row, image_column] = np.dot(image_vector, filter_vector)
    return output


def convolution_backward(parameters, gradients, image_volume):
    """ Calculate gradient of loss with respect to convolution filter weights """
    image_depth, image_height, image_width = image_volume.shape
    filter_count, filter_depth, filter_height, filter_width = parameters.shape
    gradient_loss_filters = np.zeros(parameters.shape)
    for filter_index, _ in enumerate(parameters):
        for image_row in range(image_height - filter_height + 1):
            for image_column in range(image_width - filter_width + 1):
                # Get 3D image region volume
                image_region = image_volume[:image_depth, image_row : image_row + filter_height, image_column : image_column + filter_width]
                # Convolve image region with gradient of loss with respect to convolved image output
                gradient_loss_filters[filter_index] += gradients[filter_index, image_row, image_column] * image_region
    return gradient_loss_filters


def convolution_update(parameters, gradients, learning_rate):
    """ Update filter weights using gradient of loss with respect to filter weights """
    return parameters - learning_rate * gradients


def maxpooling_forward(image_volume, pooling_size):
    """ Perform maximum pooling operation on a 3D image volume """
    image_depth, image_height, image_width = image_volume.shape
    pooled_height = image_height // 2
    pooled_width = image_width // 2
    image_region = np.zeros((pooling_size, pooling_size))
    output = np.zeros((image_depth, pooled_height, pooled_width))
    for image_index, image in enumerate(image_volume):
        for pooled_row in range(pooled_height):
            for pooled_column in range(pooled_width):
                # Get 2D image region
                image_region = image[
                    pooled_row * pooling_size : pooling_size * (pooled_row + 1),
                    pooled_column * pooling_size : pooling_size * (pooled_column + 1)
                ]
                # Find maximum pixel value in region
                output[image_index, pooled_row, pooled_column] = np.amax(image_region)
    return output


def maxpooling_backward(gradients, image_volume, pooling_size):
    """ Calculate gradient of loss with respect to max pooling input image """
    image_depth, image_height, image_width = image_volume.shape
    pooled_height = image_height // 2
    pooled_width = image_width // 2
    image_region = np.zeros((pooling_size, pooling_size))
    output = np.zeros((image_depth, image_height, image_width))
    for image_index, image in enumerate(image_volume):
        for image_row in range(pooled_height):
            for image_column in range(pooled_width):
                # Get 2D image region
                image_region = image[
                    image_row * pooling_size : pooling_size * (image_row + 1),
                    image_column * pooling_size : pooling_size * (image_column + 1),
                ]
                # Find maximum pixel value in region
                maximum_pixel = np.amax(image_region)
                for pooled_row in range(pooling_size):
                    for pooled_column in range(pooling_size):
                        # Get pixel location containing maximum value
                        if image_region[pooled_row, pooled_column] == maximum_pixel:
                            # Only update found location with gradient of loss with respect to max pooling output
                            output[
                                image_index,
                                image_row * pooling_size + pooled_row,
                                image_column * pooling_size + pooled_column
                            ] = gradients[image_index, image_row, image_column]
                            # Once one update has been made, don't need to update others
                            break
    return output


def dense_forward(parameters, image_volume):
    ''' Pass linear combination of data and weights through activation function '''
    # Convert image volume into 1D vector
    image_vector = image_volume.flatten()
    weights, biases = parameters
    # Calculate linear combination of network parameters and input
    activations = np.dot(image_vector, weights) + biases
    output = softmax(activations)
    return output, activations


def dense_backward(parameters, gradients, image_volume, activations):
    """ Calculate gradient of loss with respect to dense layer input """
    weights, biases = parameters
    for index, gradient in enumerate(gradients):
        # Results will remain zero if gradient of loss with respect to outputs is zero
        if gradient == 0:
            continue
        activations_exponential = np.exp(activations)
        activations_sum = np.sum(activations_exponential)
        gradient_outputs_activations = -activations_exponential[index] * activations_exponential / (activations_sum ** 2)
        gradient_outputs_activations[index] = activations_exponential[index] * (activations_sum - activations_exponential[index]) / (activations_sum ** 2)
        gradient_loss_activations = gradient * gradient_outputs_activations
        gradient_loss_inputs = weights @ gradient_loss_activations
    return gradient_loss_inputs.reshape(image_volume.shape), gradient_loss_activations


def dense_update(parameters, gradient_loss_activations, image_volume, learning_rate):
    """ Update parameters using gradient of loss with respect to weights and biases """
    weights, biases = parameters
    gradient_loss_weights = image_volume.flatten()[np.newaxis].T @ gradient_loss_activations[np.newaxis]
    gradient_loss_biases = gradient_loss_activations
    weights_updated = weights.copy()
    biases_updated = biases.copy()
    weights_updated -= learning_rate * gradient_loss_weights
    biases_updated -= learning_rate * gradient_loss_biases
    return (weights_updated, biases_updated)


def main(data_train, data_test, image_shape, convolution_shape, convolution_stride, pooling_size, learning_rate):
    """ Orchestrate training and testing of convolutional network """
    np.random.seed(1)
    print('Reading training data...')
    features_train, labels_train = get_data(data_train)
    print('Reading testing data...')
    features_test, labels_test = get_data(data_test)
    print('Preparing data...')
    features_train = prepare_features(features_train, image_shape)
    features_test = prepare_features(features_test, image_shape)
    print('Initialising network...')
    output_shape = len(np.unique(labels_train))  # number of classes
    network = create_network(
        convolution_shape,
        convolution_stride,
        pooling_size,
        image_shape,
        output_shape
    )
    network, outputs, losses, true_positives = train_network(
        network,
        features_train,
        labels_train,
        image_shape,
        pooling_size,
        learning_rate
    )
    outputs, losses, true_positives = test_network(
        network,
        features_test,
        labels_test
    )


if __name__ == "__main__":
    main(
        data_train,
        data_test,
        image_shape,
        convolution_shape,
        convolution_stride,
        pooling_size,
        learning_rate
    )
