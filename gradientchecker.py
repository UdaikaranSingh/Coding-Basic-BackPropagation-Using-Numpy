import neuralnet
import numpy as np

def gradient_check(loss_func, weight, gradient, epsilon):     # can also put in a bias in the weight input
    epsilon_squared = np.power(epsilon)
    approx_grad = (loss_func(weight + epsilon) - loss_func(weight - epsilon) ) / (2 * epsilon)
    if abs(approx_grad - gradient) > epsilon_squared:
        return 'Gradient is incorrect'
    return 'Gradient is correct'


    # can also do a forward pass on E(w + e) and E(w - e) before doing this computation
