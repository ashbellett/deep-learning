# File path of input data
file_name = "/home/ash/code/deep-learning/data/fashion.csv"

# Image dimensions
image_width = 28
image_height = 28

# Architecture of convolutional network
layers = (
  {
    'type': 'convolution',
    'properties': {
      'shape': (8,3,3),
      'stride': 1,
      'learning_rate': 0.005,
      'activation': 'relu'
    }
  }, {
    'type': 'maxpooling',
    'properties': {
      'size': 2,
      'stride': 2
    }
  }, {
    'type': 'dense',
    'properties': {
      'learning_rate': 0.01,
      'activation': 'softmax'
    }
  }
)

# Number of samples to use in test set
test_size = 1000
