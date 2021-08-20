"""NEXT STEP: HOW TO PERFORM BACKPROPAGATION"""

import numpy as np
import random


class Neuron(object):
	def __init__(self, numWeights):
		self.weights = [random.random() for i in range(numWeights)]
		self.bias = random.random()
		
	def dotProduct(self, x: list) -> list:
		# x = [1,2,3]
		assert len(x) == len(self.weights) # Make sure we have the same number of weights as input data
		output = 0
		for i, weight in enumerate(self.weights):
			partial = weight * x[i]
			output += partial
		output += self.bias
		return output

	def __len__(self):
		return len(self.weights)

class Layer(object):
	"""docstring for Layer"""
	def __init__(self, numNeurons):
		self.neurons = [Neuron(numWeights=3) for n in range(numNeurons)]

	def forward(self, x):
		out = []
		for n in self.neurons:
			neuronOutput = n.dotProduct(x)
			out.append(neuronOutput)

		return out

	def __len__(self):
		return len(self.neurons)

class Network(object):
	def __init__(self):
		self.fc1 = Layer(3)
		self.fc2 = Layer(3)
		self.outputLayer = Layer(1)
		self.layers = [self.fc1, self.fc2, self.outputLayer]

	def forward(self, x):
		for l in self.layers:
			x = l.forward(x)
		
		return x[0]

	def __len__(self):
		return len(self.layers)

	def loss(self, output, truth):
		return (output - truth)**2 # Squared Error


def main():
	x = [1,2,3]
	y = 5.0
	n = Network()
	output = n.forward(x)
	loss = n.loss(output, y)
	print(f'Loss: {loss}\nOutput:{output}')

if __name__ == '__main__':
	main()

	