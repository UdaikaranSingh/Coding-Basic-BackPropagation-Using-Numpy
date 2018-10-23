import neuralnet
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def main():

	train_data_fname = 'MNIST_train.pkl'
	valid_data_fname = 'MNIST_valid.pkl'
	test_data_fname = 'MNIST_test.pkl'
	X_train, y_train = neuralnet.load_data(train_data_fname)
	X_valid, y_valid = neuralnet.load_data(valid_data_fname)
	X_test, y_test = neuralnet.load_data(test_data_fname)


	# found this as the optimal number of epochs from Part C
	# optimal value is about 26 epochs
	# To test regularization, going to check for a few more epochs
	neuralnet.config['epochs'] = 30

	regularization_constant_testers = [0.0001, 0.001]


	for regFactor in regularization_constant_testers:

		neuralnet.config['L2_penalty'] = regFactor
		network = neuralnet.Neuralnetwork(neuralnet.config)

		training_errors, validation_errors, best_model, numEpochs = neuralnet.trainer(network, X_train, y_train, X_valid, y_valid, network.config)
		
		network.layers = best_model
		accuracy = neuralnet.test(network, X_test, y_test, network.config)

		print("Regularization Constant: ", regFactor)
		print("Accuracy on Test Set: ", accuracy)
		print()
		
		plt.plot(range(len(training_errors)), training_errors,"ro", color = "blue", label='Training Error')
		plt.plot(range(len(validation_errors)), validation_errors,"ro", color = "red", label='Validation Error')
		plt.legend(loc='upper left')
		plt.xlabel("Epochs")
		plt.ylabel("Percentage Correct")
		plt.title("Training with regularization factor: " + str(regFactor))
		name = "partD_" + str(regFactor) + ".png"
		plt.savefig(name)
		plt.close()
		

		
if __name__ == '__main__':
  main()