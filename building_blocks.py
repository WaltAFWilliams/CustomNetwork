"""NEXT STEP: FIX ERROR WITH FORWARD PROPAGATION (NUMBER OF WEIGHTS IN OUTPUT LAYER MUST MATCH NUMBER OF NODES IN HIDDEN"""

import numpy as np
import random

def sigmoidDerivative(x):
	return x * (1.0 - x) # Derivative of sigmoid function for computing gradients

class Neuron():
	def __init__(self, numWeights):
		self.weights = [random.random() for i in range(numWeights)] # All weights initialized in [0, 1]
		self.bias = random.random()
		self.error = 0.0 # error for backpropagation
		self.output = 0.0
		self.delta = 0.0
		
	def dotProduct(self, x):
		# x = [1,2,3]
		assert len(x) == len(self.weights) # Make sure we have the same number of weights as input data
		output = 0
		for i, weight in enumerate(self.weights):
			output += weight * x[i]
		output += self.bias
		return output

	def __len__(self):
		return len(self.weights)

	def activate(self, output):
		self.output = 1 / (1+np.exp(-(output))) # sigmoid function
		return self.output

class Layer():
	def __init__(self, numNeurons, numWeights):
		self.neurons = [Neuron(numWeights=numWeights) for n in range(numNeurons)]

	def forward(self, x):
		out = []
		for n in self.neurons:
			linear_transform = n.dotProduct(x)
			activation = n.activate(linear_transform)
			n.output = activation
			out.append(activation)
		return out

	def __getitem__(self, idx):
		return self.neurons[idx]

	def __len__(self):
		return len(self.neurons)

class Network():
	def __init__(self, inputs, labels):
		self.inputs = inputs
		self.labels = labels
		self.fc1 = Layer(numNeurons=len(inputs), numWeights=len(inputs))
		self.outputLayer = Layer(numNeurons=len(labels), numWeights=len(self.fc1.neurons[0]))
		self.layers = [self.fc1, self.outputLayer]
	
	def forward(self, x):
		for l in self.layers:
			x = l.forward(x)
		return x

	def computeErrors(self):
		for i in reversed(range(len(self.layers))): # Need to iterate backwards through network
			errors = []
			layer = self.layers[i]
			if i == len(self.layers)-1: # output layer
				for j in range(len(layer)):
					neuron = layer[j]
					error = (neuron.output - self.labels[j]) 
					neuron.error = error 
					errors.append(2 * error) # Need to multiply by 2 because using sum square error as loss function
			
			else:
				for j in range(len(layer)): # hidden layers
					error = 0.0
					neuron = layer[j]
					for outputNeuron in self.layers[i+1]: # aggregate errors from output layer
						error += outputNeuron.error
					errors.append(error)
			
			for x in range(len(layer)):
				neuron = layer[x]
				neuron.delta = errors[x] * sigmoidDerivative(neuron.output) # will need to multiply by hidden layer outputs to update hidden parameters

	def updateWeights(self, lr=0.001):
		for i in range(len(self)):
			pass
		# for l in range(len(self.layers)):
			# for neuron in l:
				# for weight in neuron.weights:
					# weight = weight + lr * neuron.delta * 
			
	def __len__(self):
		return len(self.layers)

	def __getitem__(self, i):
		return self.layers[i]

	def loss(self, outputs, gts):
		out = []
		for i, num in enumerate(outputs): # Sum Squared Error
			out.append((num - gts[i])**2) 
		
		return out

