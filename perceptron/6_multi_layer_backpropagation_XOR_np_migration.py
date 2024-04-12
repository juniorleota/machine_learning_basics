"""
This class will explore using np to simplify alot of the calculations done for MLP.
"""

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy_loss(predicted_output, expected_output):
    epsilon = 1e-10
    # this ensures that there is no log(0), so the first filter is to ensure that this is not exactly 1 since we have 1 - predicted_output in formula
    predicted_output_clipped = min(predicted_output, 1 - epsilon)
    # this ensures that pred_output is never 0
    predicted_output_clipped = max(predicted_output_clipped, epsilon)
    return -1 * (
        expected_output * np.log(predicted_output_clipped)
        + (1 - expected_output) * np.log(1 - predicted_output_clipped)
    )
    pass


def cross_entropy_loss_derivative(predicted_output, expected_output):
    return -(expected_output / predicted_output) + ((1 - expected_output) / (1 - predicted_output))


class MLP:
    def __init__(self, epochs=1000, learning_rate=0.1):
        self.epochs = epochs
        self.lr = learning_rate
        # row is hidden neuron, col is input
        self.w_input_to_hidden = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.b_hidden = np.zeros(2)
        # row is output neuron, col is hidden neuron
        self.w_hidden_to_output = np.array([0.25, 0.45])
        self.b_output = np.zeros(1)

    def forw_pass(self, input_vector):
        hidden_output = np.dot(self.w_input_to_hidden, input_vector) + self.b_hidden
        hidden_activation = sigmoid(hidden_output)
        output_output = (
            np.dot(hidden_activation, self.w_hidden_to_output) + self.b_output
        )
        output_activation = sigmoid(output_output)
        return (hidden_output, hidden_activation, output_output, output_activation)

    def back_pass(
        self,
        input_vector,
        expected_output,
        hidden_output,
        hidden_activation,
        output_output,
        output_activation,
    ):
        '''
        There are 2 pathsway for each layer:
        - change in loss for change in weight
        - change in loss for change in bias
        Now this is more relevant in last layer but we can use chain rule to connect it all
        the way through
        '''
        # change in loss wrt ouput activation
        # change in ouput activation wrt output output 
        # change in ouput output wrt hidden2ouput weights
        # change in ouput output wrt output bias 
        # change in ouput output wrt hidden activation 
        # change in hidden activation wrt hidden ouput
        # change in hidden output wrt to input2hidden weights
        # change in hidden output wrt to hidden bias
        pass

    def train(self, training_data, labels):
        pass


if __name__ == "__main__":
    training_data = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    labels = np.array([0, 0, 1, 1])
    mlp = MLP()
    res = mlp.forw_pass(np.array([1, 1]))
    print(res)
