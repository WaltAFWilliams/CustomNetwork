import numpy as np
import pandas as pd
from building_blocks import CustomNetwork

def main():
	df = pd.read_csv('~/uofm/4151/data/iris.csv')
	df = df[df['Species'].isin(['virginica', 'versicolor'])].sample(frac=1) # cut setosa labels and shuffle dataframe
	df['labels'] = df.apply(lambda x: int(x['Species']=='virginica'), axis=1) # convert labels from strings to bits
	df = df.drop('Species', axis=1)
	inputs = df.drop('labels', axis=1).values.tolist()
	labels = df['labels'].values.tolist()
	for x in range(len(labels)): # Convert single bit into one hot encoded list ( 0 -> [0,1])
		labels[x] = [labels[x], int(not labels[x])]

	n = CustomNetwork(inputs=inputs, labels=labels)
	n.fit(epochs=1000)

if __name__ == '__main__':
	main()