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


	#found this as the optimal number of epochs from Part C
	neuralnet.config['epochs'] = 36

	regularization_constant_testers = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


	for regFactor in regularization_constant_testers:
		neuralnet.config['L2_penalty'] = regFactor
		network = neuralnet.Neuralnetwork(neuralnet.config)

		training_error, validation_error, best_model = neuralnet.trainer(network, X_train, y_train, X_valid, y_valid, network.config)
		
		network.layers = best_model
		accuracy = neuralnet.test(network, X_test, y_test, network.config)

		print("Regularization Constant: ", regFactor)
		print("Accuracy on Test Set: ", accuracy)
		print()
		
		plt.plot(range(len(training_errors)), training_errors,"ro", color = "blue")
		plt.plot(range(len(validation_errors)), validation_errors,"ro", color = "red")
		plt.set_xlabel("Epochs")
		plt.set_ylabel("Percentage Correct")
		plt.set_title("Training with regularization factor: " + regFactor)
		name = "partD_" + str(regFactor) + ".png"
		plt.savefig(name)

		
if __name__ == '__main__':
  main()