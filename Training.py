#!/usr/bin/env python

"""
Lasagne implementation of ILSVRC2015 winner on the mnist dataset
Deep Residual Learning for Image Recognition
http://arxiv.org/abs/1512.03385
"""

from __future__ import print_function

import sys
import os
import time
import string
import pickle

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.layers.dnn
import BatchNormLayer
sys.setrecursionlimit(10000)

# ##################### Build the neural network model #######################

def build_cnn(input_var=None, n=1, num_filters=8):
    # Setting up layers
#    conv = lasagne.layers.Conv2DLayer
    conv = lasagne.layers.dnn.Conv2DDNNLayer # cuDNN
    nonlinearity = lasagne.nonlinearities.rectify
    sumlayer = lasagne.layers.ElemwiseSumLayer
#    scaleandshiftlayer = parmesan.layers.ScaleAndShiftLayer
#    normalizelayer = parmesan.layers.NormalizeLayer
    batchnorm = BatchNormLayer.batch_norm
    # Conv layers must have batchnormalization and
    # Micrsoft PReLU paper style init(might have the wrong one!!)
    def convLayer(l, num_filters, filter_size=(1, 1), stride=(1, 1),
                  nonlinearity=nonlinearity, pad='same', W=lasagne.init.HeNormal(gain='relu')):
        l = conv(l, num_filters=num_filters, filter_size=filter_size,
            stride=stride, nonlinearity=nonlinearity,
            pad=pad, W=W)
        # Notice that the batch_norm layer reallocated the nonlinearity form the conv
        l = batchnorm(l)
        return l
    
    # Bottleneck architecture as descriped in paper
    def bottleneckDeep(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    # Bottleneck architecture with more efficiency (the post with Kaiming he's response)
    # https://www.reddit.com/r/MachineLearning/comments/3ywi6x/deep_residual_learning_the_bottleneck/
    def bottleneckDeep2(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, nonlinearity=nonlinearity)
        return l
    # The "simple" residual block architecture
    def bottleneckShallow(l, num_filters, stride=(1, 1), nonlinearity=nonlinearity):
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), stride=stride, nonlinearity=nonlinearity)
        l = convLayer(
            l, num_filters=num_filters*4, filter_size=(3, 3), nonlinearity=nonlinearity)
        return l
        
    bottleneck = bottleneckShallow

    # Simply stacks the bottlenecks, makes it easy to model size of architecture with int n   
    def bottlestack(l, n, num_filters):
        for _ in range(n):
            l = sumlayer([bottleneck(l, num_filters=num_filters), l])
        return l

    # Building the network
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 3, 7),
                                        input_var=input_var)
    # First layer just a plain convLayer
    l1 = convLayer(
	    l_in, num_filters=num_filters*4, filter_size=(3, 3)) # Filters multiplied by 4 as bottleneck returns such size

    # Stacking bottlenecks and making residuals!

    l1_bottlestack = bottlestack(l1, n=n-1, num_filters=num_filters) #Using the -1 to make it fit with size of the others
    l1_residual = convLayer(l1_bottlestack, num_filters=num_filters*4*2, stride=(2, 2), nonlinearity=None) #Multiplying by 2 because of feature reduction by 2

    l2 = sumlayer([bottleneck(l1_bottlestack, num_filters=num_filters*2, stride=(2, 2)), l1_residual])
    l2_bottlestack = bottlestack(l2, n=n, num_filters=num_filters*2)
    l2_residual = convLayer(l2_bottlestack, num_filters=num_filters*2*2*4, stride=(2, 2), nonlinearity=None)# again, this is now the second reduciton in features

    l3 = sumlayer([bottleneck(l2_bottlestack, num_filters=num_filters*2*2, stride=(2, 2)), l2_residual])
    l3_bottlestack = bottlestack(l3, n=n, num_filters=num_filters*2*2)

    # And, finally, the 10-unit output layer:
    network = lasagne.layers.DenseLayer(
            l3_bottlestack,
            num_units=5,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(n=1, num_filters=8, num_epochs=5):
    assert n>=0
    assert num_filters>0
    assert num_epochs>0
    print("Amount of bottlenecks: %d" % n)
    # Load the dataset
    print("Loading data...")
    namei = open('Inputs.pkl')
    nameo = open('Outputs.pkl')

    Inputs = pickle.load(namei)
    Outputs = pickle.load(nameo)

    X_train = Inputs
    y_train = Outputs
    X_val = Inputs[-500:]
    y_val = Outputs[-500:]
    X_test = Inputs[-200:]
    y_test = Outputs[-200:]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var, n, num_filters)
    all_layers = lasagne.layers.get_all_layers(network)
    num_params = lasagne.layers.count_params(network)
    num_conv = 0
    num_nonlin = 0
    num_input = 0
    num_batchnorm = 0
    num_elemsum = 0
    num_dense = 0
    num_unknown = 0
    print("  layer output shapes:")
    for layer in all_layers:
	name = string.ljust(layer.__class__.__name__, 32)
	print("    %s %s" %(name, lasagne.layers.get_output_shape(layer)))
	if "Conv2D" in name:
	    num_conv += 1
	elif "NonlinearityLayer" in name:
	    num_nonlin += 1
	elif "InputLayer" in name:
	    num_input += 1
	elif "BatchNormLayer" in name:
	    num_batchnorm += 1
	elif "ElemwiseSumLayer" in name:
	    num_elemsum += 1
	elif "DenseLayer" in name:
	    num_dense += 1
	else:
	    num_unknown += 1
    print("  no. of InputLayers: %d" % num_input)
    print("  no. of Conv2DLayers: %d" % num_conv)
    print("  no. of BatchNormLayers: %d" % num_batchnorm)
    print("  no. of NonlinearityLayers: %d" % num_nonlin)
    print("  no. of DenseLayers: %d" % num_dense)
    print("  no. of ElemwiseSumLayers: %d" % num_elemsum)
    print("  no. of Unknown Layers: %d" % num_unknown)
    print("  total no. of layers: %d" % len(all_layers))
    print("  no. of parameters: %d" % num_params)

    prediction = lasagne.layers.get_output(network)
    loss = (prediction - target_var)**2
    loss = loss.mean()

    params = lasagne.layers.get_all_params(network, trainable=True)
    learning_rate=0.0001
    momentum=0.99
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=momentum)
    updates = lasagne.updates.adadelta(loss, params, learning_rate=learning_rate, rho=momentum, epsilon=1e-7)
    #updates = lasagne.updates.adadelta(loss, params, learning_rate=learning_rate, rho=0.98, epsilon=1e-6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = (test_prediction - target_var)**2
    test_loss = test_loss.mean()

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], test_loss)

    with np.load('model27.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train , 1000, shuffle=True):
    	    inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 100, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
        test_err = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, 100, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            test_err += err
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        if epoch == 0:
            acc = val_err / val_batches
        if val_err / val_batches < acc:
            acc = val_err / val_batches
            np.savez('model_best', *lasagne.layers.get_all_param_values(network))
        if epoch%100 == 0 and epoch !=0:
            model_name = 'model' + str(epoch/100) + '.npz'
            np.savez(model_name, *lasagne.layers.get_all_param_values(network))
        accuracyfile = open('accuracy.txt','a')
        accuracyfile.write(str(train_err / train_batches) + ' ' + str(val_err / val_batches) + '\n' )
        accuracyfile.close()


    # After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 100, shuffle=False):
        inputs, targets = batch
        err = val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")

    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    # Optionally, you could now dump the network weights to a file like this:
    #np.savez('models.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual neural network on MNIST using Lasagne.")
        print("Usage: %s [NUM_BOTTLENECKS] [NUM_FILTERS] [EPOCHS]" % sys.argv[0])
        print()
        print("NUM_BOTTLENECKS: Define amount of bottlenecks with integer, e.g. 3")
	print("NUM_FILTERS: Defines the amount of filters in the first layer(doubled at each filter halfing)")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['n'] = int(sys.argv[1])
	if len(sys.argv) > 2:
	    kwargs['num_filters'] = int(sys.argv[2])
        if len(sys.argv) > 3:
            kwargs['num_epochs'] = int(sys.argv[3])
        main(**kwargs)
