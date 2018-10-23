import neuralnet
import numpy as np
import matplotlib as plt

def grad_diff(grad, approx_grad):
    return abs(grad - approx_grad)


def main():
    train_data_fname = 'MNIST_train.pkl'

    X_train, y_train = neuralnet.load_data(train_data_fname)
    epsilon = 0.1
    epsilon_squared = np.power(0.01, 2)
    nnet = neuralnet.Neuralnetwork(neuralnet.config)

    gradients = []    # will store lists of gradients and approximate gradients
    all_correct = True

    nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])

    num_layers = len(nnet.layers)

    for i in range(400):       # run the network on this training example
        nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
        nnet.backward_pass()
        for layer in nnet.layers:
            if isinstance(layer, neuralnet.Layer):
                layer.w = layer.w + 0.001 * layer.d_w
                layer.b = layer.b + 0.001 * layer.d_b


    # Check the gradient
    j = 0
    for layer in nnet.layers:  # add and subtract epsilon to the weights and do forward_pass to find E(w + e) and E(w - e)
        if isinstance(layer, neuralnet.Layer):
            if j == 0:    # input to hidden Layer
                original_w1 = layer.w[0][0]     # save original weight
                layer.w[0][0] = layer.w[0][0] + epsilon     # add epsilon and compute loss
                input_to_hidden_w1_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.w[0][0] = layer.w[0][0] - epsilon    # subtract epsilon and compute loss
                input_to_hidden_w1_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad1 = (input_to_hidden_w1_plus_loss - input_to_hidden_w1_minus_loss) / (2 * epsilon)    # approx_grad = E(w + e) - E(w - e) / 2e
                layer.w[0][0] = original_w1    # set weight back to original weight
                nnet.backward_pass()       # back pass to find gradient
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == 0:
                        grad = each_layer.d_w[0][0]     # find the gradient
                if grad_diff(grad, approx_grad1) > epsilon_squared:    # if gradients differ by episilon squared, the gradient is incorrect
                    all_correct = False
                    print('Input to hidden gradient is incorrect')
                gradients.append([grad, approx_grad1])     # append the back pass gradient and the approximate gradient as a list to gradients

                original_w2 = layer.w[0][1]
                layer.w[0][1] = layer.w[0][1] + epsilon
                input_to_hidden_w2_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.w[0][1] = layer.w[0][1] - epsilon
                input_to_hidden_w2_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad2 = (input_to_hidden_w2_plus_loss - input_to_hidden_w2_minus_loss) / (2 * epsilon)
                layer.w[0][1] = original_w2
                nnet.backward_pass()
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == 0:
                        grad = each_layer.d_w[0][1]
                if grad_diff(grad, approx_grad2) > epsilon_squared:
                    all_correct = False
                    print('Input to hidden gradient is incorrect')
                gradients.append([grad, approx_grad2])

                original_w3 = layer.b[0][0]
                layer.b[0][0] = layer.b[0][0] + epsilon
                hidden_bias_w_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.b[0][0] = layer.b[0][0] - epsilon
                hidden_bias_w_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad3 = (hidden_bias_w_plus_loss - hidden_bias_w_minus_loss) / (2 * epsilon)
                layer.b[0][0] = original_w3
                nnet.backward_pass()
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == 0:
                        grad = each_layer.d_b[0][0]
                if grad_diff(grad, approx_grad3) > epsilon_squared:
                    all_correct = False
                    print('Hidden bias gradient is incorrect')
                gradients.append([grad, approx_grad3])

            if j == num_layers - 1:        # hidden layer to output layer
                original_w4 = layer.w[0][0]
                layer.w[0][0] = layer.w[0][0] + epsilon
                hidden_to_output_w1_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.w[0][0] = layer.w[0][0] - epsilon
                hidden_to_output_w1_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad4 = (hidden_to_output_w1_plus_loss - hidden_to_output_w1_minus_loss) / (2 * epsilon)
                layer.w[0][0] = original_w4
                nnet.backward_pass()
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == num_layers - 1:
                        grad = each_layer.d_w[0][0]
                if grad_diff(grad, approx_grad4) > epsilon_squared:
                    all_correct = False
                    print('Hidden to output gradient is incorrect')
                gradients.append([grad, approx_grad4])

                original_w5 = layer.w[0][1]
                layer.w[0][1] = layer.w[0][1] + epsilon
                hidden_to_output_w2_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.w[0][1] = layer.w[0][1] - epsilon
                hidden_to_output_w2_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad5 = (hidden_to_output_w2_plus_loss - hidden_to_output_w2_minus_loss) / (2 * epsilon)
                nnet.backward_pass()
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == num_layers - 1:
                        grad = each_layer.d_w[0][1]
                if grad_diff(grad, approx_grad5) > epsilon_squared:
                    all_correct = False
                    print('Hidden to output gradient is incorrect')
                gradients.append([grad, approx_grad5])

                original_w6 = layer.b[0][6]
                layer.b[0][6] = layer.b[0][6] + epsilon
                output_bias_w_plus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                layer.b[0][6] = layer.b[0][6] - epsilon
                output_bias_w_minus_loss = nnet.forward_pass(X_train[0].reshape(1,784), y_train[0])[0]
                approx_grad6 = (output_bias_w_plus_loss - output_bias_w_minus_loss) / (2 * epsilon)
                layer.b[0][6] = original_w6
                nnet.backward_pass()
                for each_layer in nnet.layers:
                    if nnet.layers.index(each_layer) == num_layers - 1:
                        grad = each_layer.d_b[0][6]
                if grad_diff(grad, approx_grad6) > epsilon_squared:
                    print(grad_diff(grad, approx_grad6))
                    all_correct = False
                    print('Output bias gradient is incorrect')
                gradients.append([grad, approx_grad6])
        j = j + 1

    if all_correct:
        print('All gradients are correct')

    print('***********************************************')
    print('Input to hidden weight 1:')
    print('Gradient approximation:', gradients[0][1])
    print('Actual gradient:', gradients[0][0])
    print('Input to hidden weight 2:')
    print('Gradient approximation:', gradients[1][1])
    print('Actual gradient:', gradients[1][0])
    print('Hidden bias weight:')
    print('Gradient approximation:', gradients[2][1])
    print('Actual gradient:', gradients[2][0])
    print('Hidden to output weight 1:')
    print('Gradient approximation:', gradients[3][1])
    print('Actual gradient:', gradients[3][0])
    print('Hidden to output weight 2:')
    print('Gradient approximation:', gradients[4][1])
    print('Actual gradient:', gradients[4][0])
    print('Output bias weight:')
    print('Gradient approximation:', gradients[5][1])
    print('Actual gradient:', gradients[5][0])


if __name__ == '__main__':
  main()
