
""" Python implementation of a feed-forward neural network. """

import os, re
import numpy as np
from tqdm import tqdm
from config import *


def softmax(data):
    """ Pass data through softmax activation function """
    return np.exp(data) / np.sum(np.exp(data), axis=0)


def cross_entropy(probability):
    """ Calculate cross-entropy loss of probability value """
    return -np.log(probability)


def get_data(file_name, length=100):
    data = {}
    regex = re.compile('[^a-zA-Z ]')
    folders = os.listdir(file_name)
    print('Reading data')
    for index, folder in enumerate(folders):
        files = os.listdir(os.path.join(file_name, folder))
        for file in tqdm(files[:20]):
            with open(os.path.join(file_name, folder, file), encoding="utf8") as text:
                line = regex.sub('', text.readline())[:length].lower()
                data[line] = index
                # {'sentence goes here': 1, 'Another sentence is here': 0}
    return data


def encode_data(data):
    vocabulary = set([word for sentence in data.keys() for word in sentence.split(' ') if len(word) > 0])
    data_length = len(data.keys())
    vocabulary_length = len(vocabulary)
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}
    targets = np.zeros((data_length, 1))
    print('Encoding data')
    for index, sentence in tqdm(enumerate(data.keys())):
        targets[index] = data[sentence]
        encodings = np.zeros((data_length, vocabulary_length, 1))
        for word in sentence.split(' '):
            if len(word) > 0:
                encodings[index, vocabulary_index[word]] = 1
    return encodings, targets, vocabulary_length


def create_network(input_shape, output_shape, hidden_length=64):
    U = np.random.randn(hidden_length, input_shape) / 1000
    W = np.random.randn(hidden_length, hidden_length) / 1000
    V = np.random.randn(output_shape, hidden_length) / 1000
    W_bias = np.zeros((hidden_length, 1))
    V_bias = np.zeros((output_length, 1))
    return (U, W, V, W_bias, V_bias)


def forward_pass(data, parameters):
    U, W, V, W_bias, V_bias  = parameters
    hidden_states = np.zeros((W.shape[0], 1))
    hidden_states = np.tanh(U@data + W@hidden_states + W_bias)
    output = V@hidden_states + V_bias
    return output, hidden_states


def backward_pass(sentence, target, parameters, probabilities, hidden_states):
    U, W, V, W_bias, V_bias  = parameters
    sentence_length = len(sentence)
    gradient_loss_output = probabilities
    gradient_loss_output[int(target[0])] -= 1
    gradient_loss_V = gradient_loss_output*hidden_states[sentence_length].T
    gradient_loss_V_bias = gradient_loss_output
    gradient_output_hidden = V
    gradient_loss_U = np.zeros(U.shape)
    gradient_loss_W = np.zeros(W.shape)
    gradient_loss_W_bias = np.zeros(W_bias.shape)
    gradient_loss_hidden = gradient_output_hidden.T@gradient_loss_output
    for index in reversed(range(sentence_length)):
        factor = gradient_loss_hidden*(1-hidden_states[index+1]**2)
        gradient_loss_U += factor*sentence[index].T
        gradient_loss_W += factor*hidden_states[index].T
        gradient_loss_W_bias += factor
        gradient_loss_hidden = W@factor
    map(
        lambda x: np.clip(x, 0, 1, x),
        [gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias]
    )
    return (gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias)


def update_network(parameters, gradients, learning_rate):
    U, W, V, W_bias, V_bias  = parameters
    gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias = gradients
    U -= learning_rate*gradient_loss_U
    W -= learning_rate*gradient_loss_W
    V -= learning_rate*gradient_loss_V
    W_bias -= learning_rate*gradient_loss_W_bias
    V_bias -= learning_rate*gradient_loss_V_bias
    return U, W, V, W_bias, V_bias


def train_network(encodings, targets, parameters, learning_rate):
    print('Training network')
    for sentence, target in tqdm(zip(encodings, targets)):
        output, hidden_states = forward_pass(sentence, parameters)
        probabilities = softmax(output)
        loss = cross_entropy(probabilities[int(target[0])])
        gradients = backward_pass(sentence, target, parameters, probabilities, hidden_states)
        parameters = update_network(parameters, gradients, learning_rate)
    return output, hidden_states, gradients, parameters


def test_network(encodings, targets, parameters):
    print('Testing network')
    true_positives = 0
    for sentence, target in tqdm(zip(encodings, targets)):
        output, _ = forward_pass(sentence, parameters)
        probabilities = softmax(output)
        loss = cross_entropy(probabilities[int(target[0])])
        true_positives += 1 if np.argmax(output) == target else 0
    return output, loss, true_positives


def main(file_name, output_length):
    np.random.seed(1)
    data = get_data(file_name)
    encodings, targets, vocabulary_length = encode_data(data)
    parameters = create_network(vocabulary_length, output_length, vocabulary_length+1)
    for _ in range(100):
        output, hidden_states, gradients, parameters = train_network(
            encodings,
            targets,
            parameters,
            learning_rate
        )
    output, loss, true_positives = test_network(encodings, targets, parameters)
    print(100*true_positives/40)


if __name__ == "__main__":
    main(
        file_name,
        output_length
    )
