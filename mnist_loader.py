'''
mnist_loader
------------
A library to load the MNIST image data.

The data files should be downloaded from here. The same URL also 
contains a description.
    http://yann.lecun.com/exdb/mnist/


Place the files in a subdirectory named 'MNIST_data'.
'''

from mnist import MNIST

#import cPickle
import gzip
import numpy as np

def load_data():
    '''
    Opens the training data files (images and labels) as well as
    the test data files and returns them in two separate objects.

    "images" is a collection of all 60,000 images and "labels" is
    the set of matching identifiers.
    '''
    mndata = MNIST('./MNIST_data')
    # Leave the files in compressed gzip format and let the reader know.
    mndata.gz = True
    tr_imgs, tr_lbls = mndata.load_training()
    # Load the 10,000 test image data items but split these off
    # into a set of 5,000 validation data and 5,000 test data
    images, labels = mndata.load_training()
    va_imgs = images[:5000]   # Take the first 5,000
    va_lbls = labels[:5000]
    te_imgs = images[-5000:]  # Take the last 5,000
    te_lbls = labels[-5000:]


    return (tr_imgs, tr_lbls, va_imgs, va_lbls, te_imgs, te_lbls)

def load_data_wrapper():
    '''
    This wrapper function calls load_data() and then converts the image
    and label data into forms that our neural network can use.

    Image data is an array of images, each image being a 1d-array of 784 
    pixel values in the range 0-254. Convert these to NumPy arrays.

    The label data is vectorized to match the output nodes of the 
    neural net. Each element of the label array becomes a 10-element 
    1d-array representing a specific digit.
    '''

    tr_images, tr_labels, va_images, va_labels, te_images, te_labels = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_images]
    training_results = [vectorized_digit(x) for x in tr_labels]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_images]
    #validation_results = [vectorized_digit(x) for x in va_labels]
    #validation_data = zip(validation_inputs, validation_results)
    validation_data = list(zip(validation_inputs, va_labels))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_images]
    test_data = list(zip(test_inputs, te_labels))
    return (training_data, validation_data, test_data)

def vectorized_digit(j):
    '''
    Takes a single digit as input, j, and creates a 10-element
    vector representing j.  The vector is all zeroes except for
    the jth element which is set to 1.
    '''
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    images, labels = load_data()
    training_data, validation_data, test_data = load_data_wrapper()
