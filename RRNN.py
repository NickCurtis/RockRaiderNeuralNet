import numpy as np

class NeuralNet(object):
	"""A simple image recognition neural net"""
	def __init__(self):
		self.layers = [None] * 9
		np.random.seed(0)
		for i in range(len(self.layers)):
			self.layers[i] = np.random.random()
			print self.layers[i]

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
			total += hiddenTotal[i] * self.layers[i+6] #Plus 6 here to align idexing

		total = sigmoid(total)
		print "Total:",total
		return total




		
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))




if __name__ == '__main__':
	nn = NeuralNet()
	inputs = [1,1]
	nn.think(inputs)
