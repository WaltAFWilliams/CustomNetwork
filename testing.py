import numpy as np
from building_blocks import *

def main():
	x = [1,2,3]
	y = [0,1]
	n = Network(inputs=x, labels=y)
	output = n.forward(x)
	loss = n.loss(output, y)
	n.computeErrors()
	# print(f'Loss: {loss}\nOutput:{output}')
	# for l in n:
	# 	for neuron in l:
	# 		print(neuron.delta)
	n.updateWeights()

if __name__ == '__main__':
	main()
	