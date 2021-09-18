import numpy as np
from building_blocks import *

def main():
	x = [1,2,3]
	y = [0,1]
	n = Network(inputs=x, labels=y)
	n.train(inputs=x, labels=y, epochs=100)

if __name__ == '__main__':
	main()
	