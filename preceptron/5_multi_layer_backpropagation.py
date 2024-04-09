import math

'''
So this implementation doesn't work but I left it here to hightligh how initializing everything to zero will lead to symmetry issue
'''

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def transpose_matrix(mat):
    res = []
    for row in zip(*mat):
        res.append(list(row))
    return res


def cross_entropy_loss(predicted_output, expected_output):
    epsilon = 1e-10
    # this ensures that there is no log(0), so the first filter is to ensure that this is not exactly 1 since we have 1 - predicted_output in formula
    predicted_output_clipped = min(predicted_output, 1 - epsilon)
    # this ensures that pred_output is never 0
    predicted_output_clipped = max(predicted_output_clipped, epsilon)
    return -(
        expected_output * math.log(predicted_output_clipped)
        + (1 - expected_output) * math.log(1 - predicted_output_clipped)
    )


def vector_add(vector_a, vector_b):
    return [x + y for x, y in zip(vector_a, vector_b)]


def vector_mult(vector_a, vector_b):
    res = 0
    for x, y in zip(vector_a, vector_b):
        res = x * y
    return res


# each entry in weights is the connection to multiple neurons
def neuron_mat_mult(weights_matrix, input_vector, layer_biases):
    return [
        neuron_vector_mult(weight, input_vector, bias)
        for weight, bias in zip(weights_matrix, layer_biases)
    ]


def neuron_vector_mult(weights, input_vector, bias):
    return vector_mult(weights, input_vector) + bias


class Perceptron:
    def __init__(self, learning_rate=0.01, epoch=100):
        self.input_2_hidden_w = [[0.1, 0.3], [0.2, 0.4]]
        self.hidden_bias = [0, 0]
        self.hidden_2_output_w = [0.25, 0.45]
        self.output_bias = 0
        self.learning_rate = learning_rate
        self.epoch = epoch

    def forward_pass(self, input_vector):
        hidden_layer_output = neuron_mat_mult(
            self.input_2_hidden_w, input_vector, self.hidden_bias
        )
        hidden_layer_activation = [sigmoid(output) for output in hidden_layer_output]
        # no need to transpose since there is only one output neuron
        output_layer_output = (
            vector_mult(self.hidden_2_output_w, hidden_layer_activation)
            + self.output_bias
        )
        output_layer_activation = sigmoid(output_layer_output)
        return {
            "hidden_layer_output": hidden_layer_output,
            "hidden_layer_activation": hidden_layer_activation,
            "output_layer_output": output_layer_output,
            "output_layer_activation": output_layer_activation,
        }

    def train(self, training_data, labels):
        for iter in range(self.epoch):
            total_loss = 0
            for input_vector, expected_output in zip(training_data, labels):
                forward_pass_res = self.forward_pass(input_vector)
                # print(forward_pass_res)
                output_activation = forward_pass_res["output_layer_activation"]
                hidden_activation = forward_pass_res["hidden_layer_activation"]

                output_loss = cross_entropy_loss(output_activation, expected_output)
                output_sigmoid_derv = sigmoid_derivative(
                    forward_pass_res["output_layer_output"]
                )
                output_err = (expected_output - output_activation) * output_sigmoid_derv
                hidden_2_output_err_gradient = [
                    nrn_activation * output_err for nrn_activation in hidden_activation
                ]
                output_bias_err_gradient = output_err

                hidden_sigmoid_dervs = [
                    sigmoid_derivative(x)
                    for x in forward_pass_res["hidden_layer_output"]
                ]
                hidden_err_propagated = [
                    weight * output_loss * derv
                    for weight, derv in zip(
                        self.hidden_2_output_w, hidden_sigmoid_dervs
                    )
                ]
                input_2_hidden_err_gradient = [
                    input * hidden_err
                    for input, hidden_err in zip(input_vector, hidden_err_propagated)
                ]
                hidden_bias_err_gradient = hidden_err_propagated

                # Update weights
                self.hidden_2_output_w = [
                    nrn_weight - (self.learning_rate * err_grad)
                    for nrn_weight, err_grad in zip(
                        self.hidden_2_output_w, hidden_2_output_err_gradient
                    )
                ]
                self.output_bias -= self.learning_rate * output_bias_err_gradient
                new_input_2_hidden_w = []
                for row in self.input_2_hidden_w:
                    new_row = [weight - (self.learning_rate * err) for weight, err in zip(row, input_2_hidden_err_gradient)]
                    new_input_2_hidden_w.append(new_row)
                self.input_2_hidden_w = new_input_2_hidden_w
                self.hidden_bias = [
                    bias - (self.learning_rate * bias_err_grd)
                    for bias, bias_err_grd in zip(
                        self.hidden_bias, hidden_bias_err_gradient
                    )
                ]
                total_loss += output_loss
            epoch_loss = output_loss/self.epoch
            print(f"Output_loss: {epoch_loss} for iteration {iter}")
            print(f"hidden_weights: {self.input_2_hidden_w}, hidden_bias: {self.hidden_bias}, output_weights: {self.hidden_2_output_w}, output: {self.output_bias}")


if __name__ == "__main__":
    training_data = [[0, 0], [1, 1], [0, 1], [1, 0]]
    labels = [1, 1, 0, 0]
    p = Perceptron(epoch=10000, learning_rate=0.1)
    p.train(training_data, labels)
    for input in training_data:
        output = p.forward_pass(input)["output_layer_activation"]
        print(f"{input} XOR = {output}")
    # print(neurons_mult(mat, vec, bias))
    # print(cross_entropy_loss(1000, 0))