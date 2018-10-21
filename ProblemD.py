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


	neuralnet.config['epochs'] = 0
	#change number of Epochs to best value found in part C (add extra 10% # of Epochs)

	regularization_constant_testers = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


	for regFactor in regularization_constant_testers:
		neuralnet.config['L2_penalty'] = regFactor
		network = neuralnet.Neuralnetwork(neuralnet.config)

		training_error, validation_error, best_model = neuralnet.trainer(network, X_train, y_train, X_valid, y_valid, network.config)
		
		network.layers = best_model

		accuracy = neuralnet.test(network, X_test, y_test, network.config)

		print("Regularization Constant: ", regFactor)
		print("Accuracy on Test Set: ", accuracy)
		
		#plot data 
		#export plot to image
		print()

		
if __name__ == '__main__':
  main()