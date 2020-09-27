
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


def get_data(file_name, length=160):
    data = {}
    files = os.listdir(file_name)
    for index, file in enumerate(files):
        with open(os.path.join(file_name, file), "r", encoding="utf8") as lines:
            for line in lines:
                text = re.sub(r'[^a-zA-Z ]', '', line)[:length].lower()
                data[text] = index
    return data


def encode_data(data):
    vocabulary = set([word for sentence in data.keys() for word in sentence.split(' ') if len(word) > 0])
    data_length = len(data.keys())
    vocabulary_length = len(vocabulary)
    vocabulary_index = {word: index for index, word in enumerate(vocabulary)}
    targets = np.zeros((data_length, 1))
    encodings = np.zeros((data_length, vocabulary_length, 1))
    for index, sentence in tqdm(enumerate(data.keys())):
        targets[index] = data[sentence]
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
    previous_hidden_states = { 0: hidden_states }
    for index, value in enumerate(data):
        # hidden_states = np.tanh(U@data + W@hidden_states + W_bias)
        hidden_states = np.tanh(U@value + W@hidden_states + W_bias)
        previous_hidden_states[index+1] = hidden_states
    output = V@hidden_states + V_bias
    return output, hidden_states, previous_hidden_states


def backward_pass(sentence, target, parameters, probabilities, hidden_states, previous_hidden_states):
    U, W, V, W_bias, V_bias  = parameters
    sentence_length = len(sentence)
    gradient_loss_output = probabilities
    gradient_loss_output[int(target[0])] -= 1
    gradient_loss_V = gradient_loss_output@previous_hidden_states[sentence_length].T[np.newaxis]
    gradient_loss_V_bias = gradient_loss_output
    gradient_output_hidden = V
    gradient_loss_U = np.zeros(U.shape)
    gradient_loss_W = np.zeros(W.shape)
    gradient_loss_W_bias = np.zeros(W_bias.shape)
    gradient_loss_hidden = gradient_output_hidden.T@gradient_loss_output
    for index in reversed(range(sentence_length)):
        factor = (gradient_loss_hidden*(1-previous_hidden_states[index+1]**2))
        gradient_loss_U += factor@sentence[index].T[np.newaxis]
        gradient_loss_W += factor@previous_hidden_states[index].T[np.newaxis]
        gradient_loss_W_bias += factor
        gradient_loss_hidden = W@factor
    map(
        lambda x: np.clip(x, -1, 1, x),
        [gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias]
    )
    return (gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias)


def update_network(parameters, gradients, learning_rate):
    U, W, V, W_bias, V_bias  = parameters
    gradient_loss_U, gradient_loss_W, gradient_loss_V, gradient_loss_W_bias, gradient_loss_V_bias = gradients
    U_updated = U - learning_rate*gradient_loss_U
    W_updated = W - learning_rate*gradient_loss_W
    V_updated = V - learning_rate*gradient_loss_V
    W_bias_updated = W_bias - learning_rate*gradient_loss_W_bias
    V_bias_updated = V_bias - learning_rate*gradient_loss_V_bias
    # print(U - U_updated)
    return U_updated, W_updated, V_updated, W_bias_updated, V_bias_updated


def train_network(encodings, targets, parameters, learning_rate):
    for sentence, target in zip(encodings, targets):
        #print('sentence:', sentence)
        output, hidden_states, previous_hidden_states = forward_pass(sentence, parameters)
        probabilities = softmax(output)
        loss = cross_entropy(probabilities[int(target[0])])
        print(loss)
        gradients = backward_pass(sentence, target, parameters, probabilities, hidden_states, previous_hidden_states)
        parameters = update_network(parameters, gradients, learning_rate)
    return output, hidden_states, gradients, parameters


def test_network(encodings, targets, parameters):
    true_positives = 0
    for sentence, target in tqdm(zip(encodings, targets)):
        output, _ = forward_pass(sentence, parameters)
        probabilities = softmax(output)
        loss = cross_entropy(probabilities[int(target[0])])
        true_positives += 1 if np.argmax(output) == target else 0
    return output, loss, true_positives


def main(file_name, output_length):
    np.random.seed(1)
    # data = get_data(file_name)
    data = {
        'good': 1,
        'bad': 0,
        'happy': 1,
        'sad': 0,
        'not good': 0,
        'not bad': 1,
        'not happy': 0,
        'not sad': 1,
        'very good': 1,
        'very bad': 0,
        'very happy': 1,
        'very sad': 0,
        'i am happy': 1,
        'this is good': 1,
        'i am bad': 0,
        'this is bad': 0,
        'i am sad': 0,
        'this is sad': 0,
        'i am not happy': 0,
        'this is not good': 0,
        'i am not bad': 1,
        'this is not sad': 1,
        'i am very happy': 1,
        'this is very good': 1,
        'i am very bad': 0,
        'this is very sad': 0,
        'this is very happy': 1,
        'i am good not bad': 1,
        'this is good not bad': 1,
        'i am bad not good': 0,
        'i am good and happy': 1,
        'this is not good and not happy': 0,
        'i am not at all good': 0,
        'i am not at all bad': 1,
        'i am not at all happy': 0,
        'this is not at all sad': 1,
        'this is not at all happy': 0,
        'i am good right now': 1,
        'i am bad right now': 0,
        'this is bad right now': 0,
        'i am sad right now': 0,
        'i was good earlier': 1,
        'i was happy earlier': 1,
        'i was bad earlier': 0,
        'i was sad earlier': 0,
        'i am very bad right now': 0,
        'this is very good right now': 1,
        'this is very sad right now': 0,
        'this was bad earlier': 0,
        'this was very good earlier': 1,
        'this was very bad earlier': 0,
        'this was very happy earlier': 1,
        'this was very sad earlier': 0,
        'i was good and not bad earlier': 1,
        'i was not good and not happy earlier': 0,
        'i am not at all bad or sad right now': 1,
        'i am not at all good or happy right now': 0,
        'this was not happy and not good earlier': 0,
    }
    #print(data)
    print('Text count:', len(data.keys()))
    encodings, targets, vocabulary_length = encode_data(data)
    print('Vocabulary length:', vocabulary_length)
    parameters = create_network(vocabulary_length, output_length)
    for _ in range(1000):
        output, hidden_states, gradients, parameters = train_network(
            encodings,
            targets,
            parameters,
            learning_rate
        )
    output, loss, true_positives = test_network(encodings, targets, parameters)
    print(true_positives)


if __name__ == "__main__":
    main(
        file_name,
        output_length
    )
