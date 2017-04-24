'''
Copyright 2017 Nick Curtis

Redistribution and use in source and binary forms, with or without modification, are 
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of
conditions and the following disclaimer in the documentation and/or other materials
provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import sys
import os
import time
import gzip
import glob
import Image

import numpy as np
import theano
import theano.tensor as T

import lasagne

#========================================================================================
#Helper Functions
#========================================================================================

# Load our tennis ball images
def load(filename):

	#data = np.zeros((1,3,32,32))
	filelist = glob.glob('tennis_images/*.jpg')
	data = np.array([np.array(Image.open(fname)) for fname in filelist])
	data=data.reshape((data.shape[0],3,32,32))
	values = np.ones(len(data))

	#print type(data)
	return data/np.float32(256),values

# Set up the convolutional nn
def buildNetwork(inputShape,inputVal = None):

	#Create a nn that looks at images of size 32x32 with 3 channels
	nn = lasagne.layers.InputLayer(
		shape=(None,inputShape[0],inputShape[1],inputShape[2]), input_var=inputVal)

	#Set the convolutional layer to have 16 filters of size 5x5
	nn = lasagne.layers.Conv2DLayer(nn,num_filters=32,filter_size=(5,5),
		nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())

	#Create a max-pooling of factor 2 in both dimensions
	nn = lasagne.layers.MaxPool2DLayer(nn, pool_size=(2,2))

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	nn = lasagne.layers.Conv2DLayer(nn, num_filters=32, filter_size=(5, 5),
		nonlinearity=lasagne.nonlinearities.rectify)

	nn = lasagne.layers.MaxPool2DLayer(nn, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
	nn = lasagne.layers.DenseLayer(lasagne.layers.dropout(nn, p=.5),
		num_units=256,nonlinearity=lasagne.nonlinearities.rectify)


    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
	nn = lasagne.layers.DenseLayer(lasagne.layers.dropout(nn, p=.5),
		num_units=10,nonlinearity=lasagne.nonlinearities.softmax)

	return nn

# # ############################# Batch iterator ###############################
# # This is just a simple helper function iterating over training data in
# # mini-batches of a particular size, optionally in random order. It assumes
# # data is available as numpy arrays. For big datasets, you could load numpy
# # arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# # own custom data iteration function. For small datasets, you can also copy
# # them to GPU at once for slightly improved performance. This would involve
# # several changes in the main program, though, and is not demonstrated here.
# # Notice that this function returns only mini-batches of size `batchsize`.
# # If the size of the data is not a multiple of `batchsize`, it will not
# # return the last (remaining) mini-batch.

# def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
# 	print inputs
# 	assert inputs == targets
# 	if shuffle:
# 	    indices = np.arange(inputs)
# 	    np.random.shuffle(indices)
# 	for start_idx in range(0, inputs - batchsize + 1, batchsize):
# 	    if shuffle:
# 	        excerpt = indices[start_idx:start_idx + batchsize]
# 	    else:
# 	        excerpt = slice(start_idx, start_idx + batchsize)

# 	    print "EXCERPT:",excerpt
# 	    yield inputs[excerpt], targets[excerpt]
# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    #print type(inputs)
    #print type(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        #print("EXCERPT:",excerpt)
        yield inputs[excerpt], targets[excerpt]


#========================================================================================
#Main
#========================================================================================


if __name__ == '__main__':

	#Load images
	data,values = load('TennisBalls.tar.gz')


	# Prepare Theano variables for inputs and targets
	input_var = T.tensor4('inputs')
	target_var = T.ivector('targets')

	#Determine the shape of input layer for the neural network here
	input_shape = data[0].shape

	#Build the neural network using the lasagne library
	nn = buildNetwork(input_shape,input_var)

	#Helper code for preparing training

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(nn)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(nn, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
	        loss, params, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(nn, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
	                                                        target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
	                  dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


	# We'll determine the input shape from the first example from the training set.
	input_shape=data[0].shape
	l_in=lasagne.layers.InputLayer(
		shape=(None,input_shape[0],input_shape[1],input_shape[2]))

	print "DATA:",data
	
	for epoch in range(10):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		#print "SHAPE:",data.shape
		train_err += train_fn(data,values)
		print train_err
		
		for batch in iterate_minibatches(data, values, 10, shuffle=True):
		    inputs, targets = batch
		    #print("SHAPE:",inputs.shape)
		    train_err += train_fn(inputs, targets)
		    train_batches += 1

	print train_err
		
	

	np.savez('layers.txt', *lasagne.layers.get_all_param_values(nn))

	'''
	with np.load('model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

	'''