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

#from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
import random

import lasagne


#========================================================================================
#Neural Network Class (The brain)
#========================================================================================


class NeuralNet(object):
	"""A simple image recognition neural net. init takes layers as an array and the
	number of input layers, the number of output layers is determined by """
	def __init__(self, l, inL):
			self.layers = l

	def getLayers(self):
		return self.layers

	def think(self,inputs):
		# Initialize an array of hidden layer totals to be sigmoided
		hiddenTotal = [0.0] * 3
		#initialize return total
		total = 0.0
		#j = 6

		#Multiply all input values by 
		for i in range(6):
			hiddenTotal[i/2] += self.layers[i] * inputs[i%2]

		#sigmoid hidden layer and get total output layer ready for sigmoid
		for i in range(3):
			hiddenTotal[i] = sigmoid(hiddenTotal[i])
			total += hiddenTotal[i] * self.layers[i+6] #Plus 6 here to align indexing

		total = sigmoid(total)
		#print "Total:",total
		#print "hidden Total:",hiddenTotal
		return total,hiddenTotal,hiddenTotal



		#POSSIBLE THIS TRAINER IS BROKEN
	def learn(self, inputs, target):
		tempWeights = [None] * 3

		activation, hiddenLayer,hiddenTotal = self.think(inputs)
		error = float(target) - activation
		deltaOutput = sigDeriv(activation) * error

		#print "OLD LAYERS:",self.layers

		for i in range(3):
			#print deltaOutput * hiddenLayer[i]
			tempWeights[i] = deltaOutput * layers[i+6] * sigDeriv(hiddenTotal[i])
			layers[i+6] += deltaOutput * hiddenLayer[i]

		for i in range(6):
			layers[i] += tempWeights[i/2] * inputs[i%2]


		#print "NEW LAYERS:",self.layers
 

		#print "DELTA OUTPUT:",deltaOutput
		print error



#========================================================================================
#Helper Functions
#========================================================================================

# Load our tennis ball images
def load():
	pass


# Set up the convolutional nn
def buildNetwork(inputVal = None):

	#Create a nn that looks at images of size 32x32 with 3 channels
	nn = lasagne.layers.InputLayer(shape=(None,3,32,32), input_var=inputVal)

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


#The activation function		
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))

#The derivative of the activation funtion, used for regression
def sigDeriv(x):
	return np.exp(x)/((1 + np.exp(x))**2)

#return an array of floats from given file
def read(file):
	#strip and cast as float for each line in file
	return [float(line.rstrip('\n')) for line in open(file)]


def write(file,values,length,init = False):
	f = open(file,'w')
	#If we wish to initialize with default values
	if init == True:
		np.random.seed(0)
		for i in range(length):
			f.write('%f\n'%(np.random.random()))

	#else write current values to file
	else:
		for i in range(len(values)):
			f.write('%f\n',values[i])
	f.close()


#========================================================================================
#Main
#========================================================================================


if __name__ == '__main__':

	nn = buildNetwork()

	np.savez('layers.txt', *lasagne.layers.get_all_param_values(nn))

	'''
	with np.load('model.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)

	'''
