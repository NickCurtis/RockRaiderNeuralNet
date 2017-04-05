import numpy as np
import sklearn

class NeuralNet(object):
	"""docstring for NeuralNet"""
	def __init__(self):
		self.layers = [None] * 8
		np.random.seed(0)
		for i in range(len(self.layers)):
			self.layers[i] = np.random.random()
			print self.layers[i]

	def think(self,inputs):
		# Initialize an array of hidden layer totals to be sigmoided
		#toHiddens = [None] * 6
		hiddenTotal = [0.0] * 3
		for i in range(6):
			hiddenTotal[i/2] += self.layers[i] * inputs[i%2]
			#print "i:",inputs[i/3]
		for i in range(3):
			hiddenTotal[i] = sigmoid(hiddenTotal[i])
			print "HIDDEN TOTAL:",hiddenTotal[i]
			'''
		hiddenTotal[0] = sigmoid(hiddenTotal[0])
		hiddenTotal[1] = sigmoid(hiddenTotal[1])
		print hiddenTotal[0],hiddenTotal[1]
		'''




		
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))




if __name__ == '__main__':
	nn = NeuralNet()
	inputs = [1,1]
	nn.think(inputs)
