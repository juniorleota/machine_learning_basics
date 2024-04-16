"""
Note:
- when doing the derivative of loss error you should ignore the summation function since you want to 
keep each data to be per input vector and not collapsed. 
"""

import numpy as np
from utils import loss_viz as lviz
from utils import nn_func as nnf


class MLP:
    def __init__(
        self,
        epochs=1000,
        learning_rate=0.1,
        input_features_size=2,
        hidden_neuron_size=4,
        show_grad=False,
    ):
        self.epochs = epochs
        self.lr = learning_rate
        # col is hidden neuron, row is input connection to neuron
        self.w_input_to_hidden = np.random.uniform(
            size=(input_features_size, hidden_neuron_size)
        )
        self.b_hidden = np.zeros(hidden_neuron_size)
        output_neurons_size = 1
        self.w_hidden_to_output = np.random.uniform(
            size=(hidden_neuron_size, output_neurons_size)
        )
        self.b_output = np.zeros(output_neurons_size)
        self.loss_viz = lviz.LossViz()
        self.show_grad = show_grad

    def forw_pass(self, training_data):
        hidden_output = np.dot(training_data, self.w_input_to_hidden) + self.b_hidden
        hidden_activation = nnf.sigmoid(hidden_output)
        output_output = (
            np.dot(hidden_activation, self.w_hidden_to_output) + self.b_output
        )
        output_activation = nnf.sigmoid(output_output)
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
        dl_dao = nnf.mse_d(output_activation, labels)
        dao_dzo = dl_dao * nnf.sigmoid_d(output_output)
        dzo_dwh2o = np.dot(hidden_activation.T, dao_dzo)
        dzo_dbo = np.sum(dao_dzo, axis=0)
        dzo_dah = np.dot(dao_dzo, self.w_hidden_to_output.T)
        dah_dzh = dzo_dah * nnf.sigmoid_d(hidden_output)
        dzh_dwi2h = np.dot(training_data.T, dah_dzh)
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
            loss = nnf.mse(output_activation, labels)
            if loss < 0.01:
                print(
                    f"loss is already at 99.999%, so stopping training at iteration {iter}"
                )
                break
            if iter % 100 == 0:
                print(f"Loss: {loss} for iteration {iter}")
                loss_data.append([iter, loss])
        if self.show_grad:
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
