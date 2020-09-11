# File path of input data
# Assumes first column is label and other columns are features
data_train = ""
data_test = ""

# Input image dimensions
# (depth, height, width)
image_shape = (1, 28, 28)

# Convolutional layer hyper-parameters
convolution_shape = (8, 3, 3)
convolution_stride = 1

# Max pooling layer hyper-parameters
pooling_size = 2

# Learning rate for gradient descent
learning_rate = 0.005
