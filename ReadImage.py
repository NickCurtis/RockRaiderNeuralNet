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

import cv2
import numpy as np
import theano
import theano.tensor as T
import time

import lasagne

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



if __name__ == '__main__':	
	
	input_var = T.tensor4('inputs')

	#Create and show this image for testing (I'm pointing the camera back at the screen
	#so it can see a tennis ball)
	img = cv2.imread('Tennis.jpg',-1)
	cv2.imshow('image',img)


	cropped = img[0:1080,420:1500]
	cropped = cv2.resize(cropped, (32, 32))

	cropped = np.array([cropped])/np.float64(256)
	cropped = cropped.reshape((cropped.shape[0],3,32,32))

	
	#Build a network with random weights
	nn = buildNetwork((3,32,32),input_var)

	#Load our network with weights from training
	with np.load('layers.txt.npz') as f:
		layers = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(nn,layers)

	#Create the prediction function to guess whether image is a tennis ball or not
	prediction = lasagne.layers.get_output(nn, deterministic=True)
	predict_fn = theano.function([input_var], prediction)


	#Make a prediction on our ideal tennis ball
	print("Predicted class for first test input: %r" % predict_fn(cropped))
	print np.median(predict_fn(cropped))
	

	'''
	#Set up camera feed
	cap = cv2.VideoCapture(0)

	
	while(True):
	    # Capture frame-by-frame
	    ret, frame = cap.read()

	    #crop the image to 32x32 from 480p
	    cropped = frame[0:480,80:560]
	    cropped = cv2.resize(cropped, (32, 32))

	    #Convert image to float for prediction function
	    cropped = np.array([cropped])/np.float32(256)
	    cropped = cropped.reshape((cropped.shape[0],3,32,32))
	    prediction = predict_fn(cropped)
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
	        break

	

	# When everything done, release the capture
	cap.release()

	'''
	
	
	cv2.destroyAllWindows()