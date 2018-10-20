import neuralnet
import numpy as np
import pickle

def main():

	train_data_fname = 'MNIST_train.pkl'
	valid_data_fname = 'MNIST_valid.pkl'
	test_data_fname = 'MNIST_test.pkl'
	X_train, y_train = neuralnet.load_data(train_data_fname)
	X_valid, y_valid = neuralnet.load_data(valid_data_fname)
	X_test, y_test = neuralnet.load_data(test_data_fname)


	config = neuralnet.config
	config['epochs'] = 100

	#test with 25 hidden units
	#track error and plot
	#save plot

	#test with 100 hidden units
	#track error and plot
	#save plot


	#test with 2 hidden layers with 25 hidden layers each
	#track error and plot
	#save plot

if __name__ == '__main__':
  main()