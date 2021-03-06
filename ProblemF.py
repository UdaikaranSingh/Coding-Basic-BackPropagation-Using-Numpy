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


	#found this as the optimal number of epochs from Part C
	neuralnet.config['epochs'] = 26

	testshapes = [[784, 25, 10], [784, 100, 10], [784, 47, 47, 10]]

	for shape in testshapes:
		neuralnet.config['layer_specs'] = shape

		network = neuralnet.Neuralnetwork(neuralnet.config)

		training_errors, validation_errors, best_model, numEpochs = neuralnet.trainer(network, X_train, y_train, X_valid, y_valid, network.config)
		
		network.layers = best_model
		accuracy = neuralnet.test(network, X_test, y_test, network.config)

		print("Shape: ", shape)
		print("Accuracy", accuracy)

		plt.plot(range(len(training_errors)), training_errors,"ro", color = "blue", label='Training Set Accuracy')
		plt.plot(range(len(validation_errors)), validation_errors,"ro", color = "red", label='Validation Set Accuracy')
		plt.legend(loc='upper left')
		plt.xlabel("Epochs")
		plt.ylabel("Percentage Correct")
		plt.title("Training with " + str(shape) + " Shape")
		name = "partF_" + str(shape) + ".png"
		plt.savefig(name)
		plt.close()

if __name__ == '__main__':
  main()