import numpy as np

class NeuralNet(object):
	"""A simple image recognition neural net"""
	def __init__(self, l = None):
		if (l == None):
			self.layers = [None] * 9
			np.random.seed(0)
			for i in range(len(self.layers)):
				self.layers[i] = np.random.random()
				print self.layers[i]
		else:
			self.layers = l

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




if __name__ == '__main__':
	nn = NeuralNet()
	inputs = [1,1]
	nn.think(inputs)
	#write('layers.txt',[],9,True)
	print read('layers.txt')
