#!/usr/bin/env python
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
import matplotlib.pyplot as plt
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

def GetPrediction(modelnumber):
	namei = open('Inputs.pkl')
	nameo = open('Outputs.pkl')

	Inputs = pickle.load(namei)
	Outputs = pickle.load(nameo)

	X_train = Inputs[:-500]
	y_train = Outputs[:-500]
	X_val = Inputs[-500:-200]
	y_val = Outputs[-500:-200]
	X_test = Inputs[-200:]
	y_test = Outputs[-200:]
	input_var = T.tensor4('inputs')
	target_var = T.matrix('targets')
	network = build_cnn(input_var, 3, 10)
	all_layers = lasagne.layers.get_all_layers(network)
	num_params = lasagne.layers.count_params(network)
	prediction = lasagne.layers.get_output(network)
	train_p = theano.function([input_var], prediction)
	modelnumber = str(modelnumber)
	if modelnumber == '0':
		modelnumber = '_best'
	loadname = 'model' + modelnumber +'.npz'
	with np.load(loadname) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
		lasagne.layers.set_all_param_values(network, param_values)

	Prediction = train_p(Inputs)*0.12 - 0.03
	return Prediction

def contrast(modelnumber = '0'):
	Pd = GetPrediction(modelnumber)
	pp1 = []
	pp2 = []
	for i in range(len(Pd))[-250:]:
		pp1.append(Pd[i][1])
	nameo = open('Outputs.pkl')
	Outputs = pickle.load(nameo)*0.06-0.03
	for i in range(len(Pd))[-250:]:
		pp2.append(Outputs[i][1])

	#**********************************************************
	fig = plt.figure(figsize=(16, 12), dpi=84, facecolor="white")
	axes = plt.subplot(111)
	axes.cla() # Clear all the information in the coordinate
	# Assign the font of the picture
	font = {'family' : 'serif', 'color'  : 'darkred', 'weight' : 'normal', 'size'   : 16}
	ax = plt.gca()
	#**********************************************************
	plt.plot(pp1)
	plt.plot(pp2)
	plt.ylabel('Log Rate')
	plt.xlabel('Days')
	plt.title('The Log Rate chart for 20150501-20160501')
	ax.grid(True)
	plt.savefig('Log Rate.png')
	plt.show()
		

def plotf(modelnumber = '0'):
	namep = open('MeanEachTenDays.pkl')
	Prediction = GetPrediction(modelnumber)
	PO = pickle.load(namep)
	P0 = []
	P1 = []
	P2 = list([PO[1]])
	P3 = list(PO[1:3])
	P4 = list(PO[1:4])
	P5 = list(PO[1:5])
	for i in range(len(PO))[30:-20]:
		P0.append(PO[i])
	for i in range(len(P0))[5:]:
		P1.append(P0[i-1]*np.exp(Prediction[i-1][0]))
		P2.append(P0[i-2]*np.exp(Prediction[i-2][1]))
		P3.append(P0[i-3]*np.exp(Prediction[i-3][2]))
		P4.append(P0[i-4]*np.exp(Prediction[i-4][3]))
		P5.append(P0[i-5]*np.exp(Prediction[i-5][4]))
	P00 = PO[:5]
	for i in range(len(P1)):
		P00.append((P1[i] + P2[i] + P3[i] + P4[i] + P5[i])/5)
	plt.plot(P0,c='y')
	plt.plot(P00)
	plt.show()	

