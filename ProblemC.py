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

	#iterate 10 times:
		#hold out 10% of the training data in each iteration
		#train the model with early stop
		#record # of epochs until the validation set had lowest error (early-stop point)
	

	#put # of epochs needed to find validation set minimum in a table
	#find the average # of epochs until validation set is at minimum

	#train new network up till the # of epochs found earlier
	#record error (on training and validation)
	#report the accruacy of model (on test set)
	#plot errors against # of epochs
	#export that plot to image



if __name__ == '__main__':
  main()