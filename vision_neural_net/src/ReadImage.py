#!/usr/bin/env python

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

#########################################################################################

#File name: ReadImage.py
#Authors: Nick

'''
Uses a convolutional neural network to detect tennis balls
'''
'''
How do you call this node?
rosrun <vision_neural_net> <ReadImage.py> <parameters>
'''

#Topics this node is subscribed to: camera feed
#Topics this node publishes to
#Services this node uses
#Other dependencies?

#########################################################################################


import cv2
import sys
import argparse
import time
import glob

import numpy as np
import theano
import theano.tensor as T
import lasagne

#CONSTANTS (organize these as necessary)
#names for constants should be in ALL CAPS


#########################################################################################


#Setup
#every node should have one
def Setup(args):


	#Set up camera feed
	if args.vid == 'cam':
		print 'Setting up camera feed...',
		cap = cv2.VideoCapture(0)
		print 'done'
	elif args.vid == 'image':
		cap = -1
	elif args.vid == 'ros':
		print 'ROS is not currently supported'
		return
	else:
		print 'Argument not supported'
		return


	print 'Building the neural network...',
	predict_fn = prepare_network()
	print 'done.'

	#Create and show this image for testing (I'm pointing the
	#camera back at the screen so it can see a tennis ball)
	#NOTE: this is a temporary solution until I can get
	#ROS to work with relative paths
	img = cv2.imread('/home/nick/catkin_ws/src/RockRaiderNeuralNet/vision_neural_net/src/Tennis.jpg',-1)
	cv2.imshow('image',img)

	#Crop the image to 32x32 so that the nn can read it
	cropped = img[0:1080,420:1500]
	cropped = cv2.resize(cropped, (32, 32))

	cropped = np.array([cropped])/np.float64(256)
	cropped = cropped.reshape((cropped.shape[0],3,32,32))

	#Make a prediction on our ideal tennis ball
	print("Predicted class for first test input: %r" % predict_fn(cropped))
	print np.median(predict_fn(cropped))

	#Run through the main loop of the program
	Loop(cap)

	cv2.destroyAllWindows()
	return
	





#Loop
#every node should have one
def Loop(cap):
	if cap == -1:
		return
	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    #crop the image to 32x32 from 480p
	    cropped_frame = frame[0:480,80:560]
	    cropped_frame = cv2.resize(cropped, (32, 32))

	    #Convert image to float for prediction function
	    cropped_frame = np.array([cropped])/np.float32(256)
	    cropped_frame = cropped_frame.reshape((cropped_frame.shape[0],3,32,32))
	    prediction = predict_fn(cropped_frame)
	    #print prediction
	    #print np.median(prediction)

	    #make a guess as to whether it's a tennis ball
	    if np.median(prediction) < 0.0000001:
	    	print "I see a tennis ball"

	    else:
	    	print "I don't see a tennis ball"

	    # Display the resulting frame
	    cv2.imshow('frame',frame)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	    	# When everything done, release the capture
	    	cap.release()
	        return




#########################################################################################


#Helper Functions

'''
parse_arguments
Allows for command line arguments to be run when calling rosrun
'''
def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("-vid", "-v",type=str, default = "cam",
	 help="Run ReadImage.py with desired video source,\
	 currently accepted parameters are 'image', 'cam', and 'ros'")
	return parser.parse_args()


'''
prepare_network
Using build_network with given weights and Theano,
set up the predictor functionused in the camera feed
'''
def prepare_network():
	input_var = T.tensor4('inputs')

	#Build a network with random weights
	nn = build_network((3,32,32),input_var)

	#Load our network with weights from training
	with np.load('/home/nick/catkin_ws/src/RockRaiderNeuralNet/vision_neural_net/src/layers.txt.npz') as f:
		layers = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(nn,layers)

	#Create the prediction function to guess whether image is a tennis ball or not
	prediction = lasagne.layers.get_output(nn, deterministic=True)
	predict_fn = theano.function([input_var], prediction)

	return predict_fn


'''
build_network
Set up the convolutional nn
'''
def build_network(inputShape,inputVal = None):

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


#########################################################################################

#Main

if __name__ == '__main__':
	args = parse_arguments()
	Setup(args)
