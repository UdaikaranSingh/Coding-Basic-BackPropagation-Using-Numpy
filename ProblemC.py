import neuralnet
import numpy as np
import pickle
import matplotlib as plt

def main():

	train_data_fname = 'MNIST_train.pkl'
	valid_data_fname = 'MNIST_valid.pkl'
	test_data_fname = 'MNIST_test.pkl'
	X_train, y_train = neuralnet.load_data(train_data_fname)
	X_valid, y_valid = neuralnet.load_data(valid_data_fname)
	X_test, y_test = neuralnet.load_data(test_data_fname)

	neuralnet.config['epochs'] = 100
	neuralnet.config['momentum'] = True

	nnet = neuralnet.Neuralnetwork(neuralnet.config)

	training_errors, validation_errors, best_model, numEpochs = neuralnet.trainer(nnet, X_train, y_train, X_valid, y_valid, nnet.config)

	print("Optimal Number of Epochs: ", numEpochs)
	print(training_errors)
	print(validation_errors)

	accuracy = neuralnet.test(nnet, X_test, y_test, nnet.config)
	print("accuracy: ", accuracy)

	#record error (on training and validation)
	#report the accruacy of model (on test set)
	#plot errors against # of epochs
	#export that plot to image



if __name__ == '__main__':
  main()