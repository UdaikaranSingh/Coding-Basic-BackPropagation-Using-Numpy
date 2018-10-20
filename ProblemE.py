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


	activation_functions = ["sigmoid", "ReLU"]

	"""
	complete this based on Part C
	Use the # of epochs found in part C
	"""

	#complete for-loop on each activation function
		#train the model
		#report accuracy of the model on test set
		#plot errors (2) to epochs
		#export plot as image


if __name__ == '__main__':
  main()