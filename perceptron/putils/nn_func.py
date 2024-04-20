import numpy as np


# this doesnt really matter an only for analysis
def cross_entropy_loss(predicted_output, expected_output):
    return -np.mean(
        expected_output * np.log(predicted_output)
        + (1 - expected_output) * np.log(1 - predicted_output)
    )


def cross_entropy_loss_derivative(predicted_output, expected_output):
    return -(expected_output / predicted_output) + (
        (1 - expected_output) / (1 - predicted_output)
    )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(predicted_output, expected_output):
    return ((expected_output - predicted_output) ** 2).mean()


def mse_d(predicted_output, expected_output):
    return -2 * (expected_output - predicted_output)


def relu(x):
    # if x > 0 then use x value
    return x * (x>0)

def relu_d(x):
    # if x > 0 then use 1 else use 0
    return 1 * (x > 0)
