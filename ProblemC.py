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

	neuralnet.config['epochs'] = 100

	nnet = neuralnet.Neuralnetwork(neuralnet.config)

	training_errors, validation_errors, best_model, numEpochs = neuralnet.trainer(nnet, X_train, y_train, X_valid, y_valid, nnet.config)

	print("Optimal Number of Epochs: ", numEpochs)

	#setting thte model to the best weights and biases
	nnet.layers = best_model
	accuracy = neuralnet.test(nnet, X_test, y_test, nnet.config)
	print("Accuracy on Test Set: ", accuracy)


	#plotting results
	plt.plot(range(len(training_errors)), training_errors, "ro", color = "blue", label='Training Set Accuracy')
	plt.plot(range(len(validation_errors)), validation_errors, "ro", color = "red", label='Validation Set Accuracy')
	plt.legend(loc='upper left')
	plt.xlabel("Epochs")
	plt.ylabel("Percentage Correct")
	plt.title("Training on MNIST Dataset")
	plt.savefig('partC.png')



if __name__ == '__main__':
  main()