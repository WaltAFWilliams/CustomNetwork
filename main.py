import numpy as np
import pandas as pd
from building_blocks import CustomNetwork

def main():
	df = pd.read_csv('iris50.csv')
	inputs = df.drop('Species', axis=1).values.tolist()
	labels = df['Species'].values.tolist()
	n = CustomNetwork(inputs=inputs, labels=labels)
	n.fit(epochs=100)

if __name__ == '__main__':
	main()
	