"""
This is just simple replacement of the loss function cross entropy to much simpler mean squared error
Note:
- when doing the derivative of loss error you should ignore the summation function since you want to 
keep each data to be per input vector and not collapsed. 
"""

import numpy as np
from utils import loss_viz as lviz

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse(predicted_output, expected_output):
    return ((expected_output- predicted_output) ** 2).mean()

# this is change in loss wrt to predicted_output
def mse_d(predicted_output, expected_output):
    # dl = mean((expected_output, pred_output)^2)
    # let u = (expected_output - pred_output)
    # l = man(u^2)
    # dl/du = u
    # du/dpred # expeceted_output(constant) - predicted_output^0 = -1
    # dl/dpred = 2(expected_ouput - pred_ouput) * -1
    # dl/dpred = mean(-2(expected_output-pred_output))
    return -2 * (expected_output - predicted_output)

class MLP:
    def __init__(self, epochs=1000, learning_rate=0.1):
        self.epochs = epochs
        self.lr = learning_rate
        # row is hidden neuron, col is input
        self.w_input_to_hidden = np.array([[0.1, 0.2], [0.3, 0.4]])
        self.b_hidden = np.zeros(2)
        # row is output neuron, col is hidden neuron
        self.w_hidden_to_output = np.array([[0.25], [0.45]])
        self.b_output = np.zeros(1)
        self.loss_viz = lviz.LossViz()

    def forw_pass(self, training_data):
        hidden_output = np.dot(training_data, self.w_input_to_hidden) + self.b_hidden
        hidden_activation = sigmoid(hidden_output)
        output_output = (
            np.dot(hidden_activation, self.w_hidden_to_output) + self.b_output
        )
        output_activation = sigmoid(output_output)
        return (hidden_output, hidden_activation, output_output, output_activation)

    def back_pass(
        self,
        training_data,
        labels,
        hidden_output,
        hidden_activation,
        output_output,
        output_activation,
    ):
        """
        There are 2 pathsway for each layer:
        - change in loss for change in weight
        - change in loss for change in bias
        The pass is : Cost <- Activation <- LinearOutput <- (Weights | Bias | Activation of Prev Layer)
        """
        # change in loss wrt ouput activation
        dl_dao = mse_d(output_activation, labels)
        # change in ouput activation wrt output output
        dao_dzo = dl_dao * sigmoid_d(output_output)
        # change in ouput output wrt hidden2ouput weights i.e the derivative of a(l-l)*w + b = a(l-1)
        # note: the order here matters
        dzo_dwh2o = np.dot(hidden_activation.T, dao_dzo)
        # change in ouput output wrt output bias, the derivative is 1
        dzo_dbo = np.sum(dao_dzo, axis=0)
        # need to figure out why this is usually np.sum(dZ2, axis=0)
        # chagne in ouput_output wrt to activation
        # z = w*a(l-1) + b => dz/da = w * a(l-1)^0 = w
        dzo_dah = np.dot(dao_dzo, self.w_hidden_to_output.T)
        # change in hidden activation wrt to hidden output
        dah_dzh = dzo_dah * sigmoid_d(hidden_output)
        # change in hidden output wrt to hidden weights
        # z = w * Input + b
        # dz/dw = Input
        dzh_dwi2h = np.dot(training_data.T, dah_dzh)
        # change in hidden output wrt to hidden bias
        # z = w * I + b
        # dz/db = 1
        dzh_dbh = np.sum(dah_dzh, axis=0)
        return dzo_dwh2o, dzo_dbo, dzh_dwi2h, dzh_dbh

    def train(self, training_data, labels):
        loss_data = []
        for iter in range(self.epochs):
            hidden_output, hidden_activation, output_output, output_activation = (
                self.forw_pass(training_data)
            )
            d_w_hidden_2_output, d_b_output, d_w_input_2_hidden, d_b_hidden = (
                self.back_pass(
                    training_data,
                    labels,
                    hidden_output,
                    hidden_activation,
                    output_output,
                    output_activation,
                )
            )
            self.w_hidden_to_output -= self.lr * d_w_hidden_2_output
            self.b_output -= self.lr * d_b_output
            self.w_input_to_hidden -= self.lr * d_w_input_2_hidden
            self.b_hidden -= self.lr * d_b_hidden
            loss = mse(output_activation, labels)
            if (loss < 0.001):
                print(f"loss is already at 99.999%, so stopping training at iteration {iter}")
                break
            if iter % 100 == 0:
                print(f"Loss: {loss} for iteration {iter}")
                loss_data.append([iter, loss])
        self.loss_viz.show(loss_data)


if __name__ == "__main__":
    training_data = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    labels = np.array([[0], [0], [1], [1]])
    epoch = 100000
    mlp = MLP(epochs=epoch, learning_rate=0.05)
    mlp.train(training_data, labels)
    for input in training_data:
        res = mlp.forw_pass(np.array([input]))[3]
        print(f"{input} = {res}")
