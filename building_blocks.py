"""NEXT STEP: HOW TO PERFORM BACKPROPAGATION"""

import numpy as np
import random

def sigmoidDerivative(x):
	return x * (1.0 - x) # Derivative of sigmoid function for computing gradients

class Neuron():
	def __init__(self, numWeights):
		self.weights = [random.random() for i in range(numWeights)] # All weights initialized in [0, 1]
		self.bias = random.random()
		self.weightGradients = [] # grads from backpropagation
		self.output = 0.0
		
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
	def __init__(self, numNeurons):
		self.neurons = [Neuron(numWeights=2) for n in range(numNeurons)]

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
	def __init__(self):
		self.fc1 = Layer(2)
		self.outputLayer = Layer(2)
		self.layers = [self.fc1, self.outputLayer]

	def forward(self, x):
		for l in self.layers:
			x = l.forward(x)
		return x

	def computeGradients(self, labels):
		for i in range(len(self.layers)-1, 0, -1): # Need to iterate backwards through network
			errors = []
			layer = self.layers[i]
			if i == len(self.layers)-1: # output layer
				for j in range(len(layer)):
					neuron = layer[j]
					error = (labels[j] - neuron.output) * sigmoidDerivative(neuron.output)  # Calculate error for output layer
					neuron.error = error
			else:
				for j in range(len(layer)): # hidden layers
					error = 0.0
					neuron = layer[j]
					for outputNeuron in self.layers[i+1]: # output layer
						error += outputNeuron.error * sigmoidDerivative(outputNeuron.output) * neuron.output
						neuron.weightGradients.append(error)	

	def __len__(self):
		return len(self.layers)

	def __getitem__(self, i):
		return self.layers[i]

	def loss(self, outputs, gts):
		out = []
		for i, num in enumerate(outputs):
			out.append(num - gts[i])
		
		return out

