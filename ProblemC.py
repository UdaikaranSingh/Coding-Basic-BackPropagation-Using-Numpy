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

	#found this as the optimal number of epochs from Part C
	neuralnet.config['epochs'] = 36

	nnet = neuralnet.Neuralnetwork(neuralnet.config)

	training_errors, validation_errors, best_model, numEpochs = neuralnet.trainer(nnet, X_train, y_train, X_valid, y_valid, nnet.config)

	print("Optimal Number of Epochs: ", numEpochs)
	print(training_errors)
	print(validation_errors)

	nnet.layers = best_model
	accuracy = neuralnet.test(nnet, X_test, y_test, nnet.config)
	print("accuracy: ", accuracy)

	plt.plot(range(len(training_errors)), training_errors,"ro", color = "blue")
	plt.plot(range(len(validation_errors)), validation_errors,"ro", color = "red")
	plt.set_xlabel("Epochs")
	plt.set_ylabel("Percentage Correct")
	plt.set_title("Training on MNIST Dataset")
	plt.savefig('partC.png')

	#record error (on training and validation)
	#report the accruacy of model (on test set)
	#plot errors against # of epochs
	#export that plot to image



if __name__ == '__main__':
  main()