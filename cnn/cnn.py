""" Python implementation of a convolutional neural network. """

import math
import numpy as np
from tqdm import tqdm
from config import *


def leaky_relu(data, derivative=False, alpha=0.001):
    ''' Pass data through leaky rectified linear unit (ReLU) activation function '''
    return np.where(data < 0, alpha*data, data) if derivative else np.where(data < 0, alpha, 1)


def softmax(data):
    ''' Pass data through softmax activation function '''
    return np.exp(data)/np.sum(np.exp(data), axis=0)


def cross_entropy(probability):
    ''' Calculate cross-entropy loss of probability value '''
    return -np.log(probability)


def get_data(file_name):
    """ Get features and labels from a CSV file """
    data = np.genfromtxt(
        file_name,
        dtype=int,
        skip_header=1,
        delimiter=","
    )
    features = data[:, 1:]/255-0.5 # {0,255} to {-0.5, 0.5}
    labels = data[:, 0]
    return (features, labels)


def split_data(data, test_size):
    ''' Split dataset into a training set and test set '''
    features, labels = data
    features_train = features[test_size:, :]
    labels_train = labels[test_size:]
    features_test = features[:test_size, :]
    labels_test = labels[:test_size]
    return (features_train, labels_train, features_test, labels_test)


def prepare_data(features, image_height, image_width):
    ''' Reshape 1D feature vectors into 2D images '''
    # Output shape is (1, image_height, image_width)
    # return np.array([image.reshape((1, image_height, image_width)) for image in features])
    return np.array([image.reshape((image_height, image_width)) for image in features])


def initialise_filters(filter_shape=(8,3,3)):
    ''' Initialise convolutional filter weights using Gaussian distribution '''
    filter_depth, filter_height, filter_width = filter_shape
    return np.random.randn(filter_depth, filter_height, filter_width)/(filter_height*filter_width)


def initialise_weights(input_nodes, output_nodes):
    ''' Initialise dense network weights using Gaussian distribution '''
    weights = np.random.randn(input_nodes, output_nodes)/input_nodes
    biases = np.zeros(output_nodes)
    return (weights, biases)


def create_network(layers, input_shape, output_shape):
    ''' Apply network configuration to initialise network weights '''
    network = []
    for layer in layers: # use enumerate here to determine length of network and then can calculate correct input nodes for dense layer
        if layer['type'] == 'convolution':
            network.append(initialise_filters(layer['properties']['shape']))
        elif layer['type'] == 'maxpooling':
            network.append(None)
        elif layer['type'] == 'dense':
            image_height, image_width = input_shape
            filter_depth, filter_height, filter_width = layers[0]['properties']['shape']
            input_nodes = filter_depth*(image_width-filter_width+1)*(image_height-filter_height+1)//4
            network.append(initialise_weights(input_nodes, output_shape))
        else:
            raise ValueError('Invalid layer type')
    return network


def forward_pass(network, layers, image_volume):
    ''' Perform forward pass of network (calculate predictions) '''
    image_volume = image_volume.copy()
    for index, (parameters, layer) in enumerate(zip(network, layers)):
        if layer['type'] == 'convolution':
            image_volume = convolution_forward(image_volume, parameters, layer['properties'])
        elif layer['type'] == 'maxpooling':
            image_volume = maxpooling_forward(image_volume, layer['properties'])
        elif layer['type'] == 'dense':
            outputs, activations = dense_forward(image_volume, parameters, layer['properties'])
        else:
            raise ValueError('Invalid layer type')
    return outputs, activations, image_volume


def backward_pass(network, layers, gradients, image_volume, activations):
    ''' Perform backward pass of network (calculate loss gradients and update network parameters) '''
    for index, (parameters, layer) in enumerate(zip(reversed(network), reversed(layers))):
        if layer['type'] == 'convolution':
            gradients = convolution_backward(gradients, image_volume, parameters)
            network[len(network)-1-index] = convolution_update(gradients, parameters, layer['properties'])
        elif layer['type'] == 'maxpooling':
            gradients = maxpooling_backward(gradients, image_volume, layer['properties'])
        elif layer['type'] == 'dense':
            gradients, gradient_loss_activations = dense_backward(gradients, image_volume, activations, parameters)
            network[len(network)-1-index] = dense_update(gradient_loss_activations, image_volume, parameters, layer['properties'])
        else:
            raise ValueError('Invalid layer type')
    return network


def train_network(network, layers, features, labels):
    ''' Orchestrate forward and backward passes of network during training '''
    network = network.copy()
    losses = []
    true_positives = []
    for iteration, (image, label) in tqdm(enumerate(zip(features, labels))):
        image_volume = image
        outputs, activations, image_volume = forward_pass(network, layers, image_volume)
        losses.append(cross_entropy(outputs[label]))
        true_positives.append(1 if np.argmax(outputs) == label else 0)
        gradients = np.zeros(10)
        gradients[label] = -1/outputs[label]
        network = backward_pass(network, layers, gradients, image_volume, activations)
        if iteration%100 == 0:
            print('Loss:', np.mean(losses))
    return network, outputs, losses, true_positives


def test_network(network, layers, features, labels):
    ''' Orchestrate forward pass of network during testing '''
    losses = []
    true_positives = []
    for iteration, (image, label) in tqdm(enumerate(zip(features, labels))):
        outputs, _, _ = forward_pass(network, layers, image)
        losses.append(cross_entropy(outputs[label]))
        true_positives.append(1 if np.argmax(outputs) == label else 0)
    return outputs, losses, true_positives


def convolution_forward(image_volume, filter_volume, properties):
    ''' Convolve a 2D image with a sequence of filters '''
    #image_depth, image_height, image_width = image_volume.shape
    image_height, image_width = image_volume.shape
    filter_depth, filter_height, filter_width,  = filter_volume.shape
    image_region = np.zeros((filter_height, filter_width))
    output = np.zeros((filter_depth, image_height-filter_height+1, image_width-filter_width+1))
    for filter_index, filter_region in enumerate(filter_volume):
        for image_row in range(image_height-filter_height+1):
            for image_column in range(image_width-filter_width+1):
                #image_region = image_volume[filter_index, image_row:image_row+filter_height, image_column:image_column+filter_width]
                image_region = image_volume[image_row:image_row+filter_height, image_column:image_column+filter_width]
                output[filter_index, image_row, image_column] = np.sum(image_region*filter_region)
    #output = leaky_relu(output)
    return output


def convolution_backward(gradients, image_volume, parameters):
    ''' Calculate gradient of loss with respect to convolution filter weights '''
    image_depth, image_height, image_width = image_volume.shape
    filter_depth, filter_height, filter_width,  = parameters.shape
    gradient_loss_filters = np.zeros(parameters.shape)
    #gradients = leaky_relu(gradients, True)
    for filter_index, filter_region in enumerate(parameters):
        for image_row in range(image_height-filter_height+1):
            for image_column in range(image_width-filter_width+1):
                image_region = image_volume[filter_index, image_row:image_row+filter_height, image_column:image_column+filter_width]
                # gradient_loss_filters[filter_index] += np.sum(gradients[filter_index, image_row, image_column]*image_region, axis=0)
                #gradient_loss_filters[filter_index] += gradients[filter_index, image_row, image_column]*image_region
                gradient_loss_filters[filter_index] = np.sum(gradients[filter_index, image_row, image_column]*image_region)
    return gradient_loss_filters


def convolution_update(gradients, parameters, properties):
    ''' Update filter weights using gradient of loss with respect to filter weights '''
    filters_updated = parameters.copy()
    filters_updated -= properties['learning_rate']*gradients
    return filters_updated


def maxpooling_forward(image_volume, properties):
    ''' Perform maximum pooling operation on a 3D image volume '''
    image_depth, image_height, image_width = image_volume.shape
    pooled_height = image_height//2
    pooled_width = image_width//2
    image_region = np.zeros((properties['size'], properties['size']))
    output = np.zeros((image_depth, pooled_height, pooled_width))
    for image_index, image in enumerate(image_volume):
        for pooled_row in range(pooled_height):
            for pooled_column in range(pooled_width):
                image_region = image[pooled_row*properties['size']:properties['size']*(pooled_row+1), pooled_column*properties['size']:properties['size']*(pooled_column+1)]
                output[image_index, pooled_row, pooled_column] = np.amax(image_region)
    return output


def maxpooling_backward(gradients, image_volume, properties):
    ''' Calculate gradient of loss with respect to max pooling input image '''
    image_depth, image_height, image_width = image_volume.shape
    pooled_height = image_height//2
    pooled_width = image_width//2
    image_region = np.zeros((properties['size'], properties['size']))
    output = np.zeros((image_depth, image_height, image_width))
    for image_index, image in enumerate(image_volume):
        for image_row in range(pooled_height):
            for image_column in range(pooled_width):
                image_region = image[image_row*properties['size']:properties['size']*(image_row+1), image_column*properties['size']:properties['size']*(image_column+1)]
                maximum_pixel = np.amax(image_region)
                for pooled_row in range(properties['size']):
                    for pooled_column in range(properties['size']):
                        if image_region[pooled_row, pooled_column] == maximum_pixel:
                            row_max_pixel, column_max_pixel = (pooled_row, pooled_column)
                            break
                output[image_index, image_row*properties['size']+row_max_pixel, image_column*properties['size']+column_max_pixel] = gradients[image_index, image_row, image_column]
    return output


def dense_forward(image_volume, parameters, properties):
    image_vector = image_volume.flatten()
    weights, biases = parameters
    activations = np.dot(image_vector, weights) + biases # add a 1 to vector to enable dot product?
    if properties['activation'] == 'relu':
        output = leaky_relu(activations)
    elif properties['activation'] == 'softmax':
        output = softmax(activations)
    else:
        raise ValueError('Invalid activation function')
    return output, activations


def dense_backward(gradients, image_volume, activations, parameters):
    weights, biases = parameters
    for index, gradient in enumerate(gradients):
        if gradient == 0:
            continue
        activations_exponential = np.exp(activations)
        activations_sum = np.sum(activations_exponential)
        gradient_outputs_activations = -activations_exponential[index]*activations_exponential/(activations_sum**2)
        gradient_outputs_activations[index] = activations_exponential[index]*(activations_sum-activations_exponential[index])/(activations_sum**2)
        gradient_loss_activations = gradient*gradient_outputs_activations
        gradient_loss_inputs = weights@gradient_loss_activations
    return gradient_loss_inputs.reshape(image_volume.shape), gradient_loss_activations


def dense_update(gradient_loss_activations, image_volume, parameters, properties):
    weights, biases = parameters
    gradient_loss_weights = image_volume.flatten()[np.newaxis].T@gradient_loss_activations[np.newaxis]
    gradient_loss_biases = gradient_loss_activations
    weights_updated = weights.copy()
    biases_updated = biases.copy()
    weights_updated -= properties['learning_rate']*gradient_loss_weights
    biases_updated -= properties['learning_rate']*gradient_loss_biases
    return (weights_updated, biases_updated)


def main(file_name, image_height, image_width, layers, test_size):
    ''' Orchestrate training and testing of convolutional network '''
    np.random.seed(1)
    data = get_data(file_name)
    features_train, labels_train, features_test, labels_test = split_data(data, test_size)
    features_train = prepare_data(features_train, image_height, image_width)
    features_test = prepare_data(features_test, image_height, image_width)
    input_shape = features_train.shape[1:] # image dimensions
    output_shape = len(np.unique(labels_train)) # number of classes
    network = create_network(layers, input_shape, output_shape)
    network, outputs, losses, true_positives = train_network(network, layers, features_train, labels_train)
    print('Network:', network)
    print('Average training loss:', np.mean(losses))
    print('Average training loss over last 1000 images:', np.mean(losses[-1000:]))
    print('Training true positive count:', np.sum(true_positives))
    print('Training true positive count over last 1000 images:', np.sum(true_positives[-1000:]))
    outputs, losses, true_positives = test_network(network, layers, features_test, labels_test)
    print('Average testing loss:', np.mean(losses))
    print('Testing true positive count:', np.sum(true_positives))


def test():
    pass


if __name__ == "__main__":
    test()
    main(file_name, image_height, image_width, layers, test_size)
