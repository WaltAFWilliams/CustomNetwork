import numpy as np
from building_blocks import *

def main():
	x = [1,2,3]
	y = [0,1]
	y = 5.0
	n = Network()
	output = n.forward(x)
	loss = n.loss(output, y)
	print(f'Loss: {loss}\nOutput:{output}')

if __name__ == '__main__':
	main()