import math


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
        self.input_2_hidden_w = [[0, 0], [0, 0]]
        self.hidden_bias = [0, 0]
        self.hidden_2_output = [0, 0]
        self.ouput_bias = 0
        self.learning_rate = learning_rate
        self.epoch = epoch

    def forward_pass(self, input_vector):
        hidden_layer_output = neuron_mat_mult(
            self.input_2_hidden_w, input_vector, self.hidden_bias
        )
        hidden_layer_activation = [sigmoid(output) for output in hidden_layer_output]
        # no need to transpose since there is only one ouput neuron
        output_layer_output = (
            vector_mult(self.hidden_2_output, hidden_layer_activation) + self.ouput_bias
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
                output_activation = forward_pass_res["output_layer_activation"]
                output_loss = cross_entropy_loss(output_activation, expected_output)
                output_sigmoid_derv = sigmoid_derivative(forward_pass_res["output_layer_output"])
                output_layer_error_gradient = (expected_output - output_activation) *  output_sigmoid_derv
                hidden_sigmoid_dervs = [sigmoid_derivative(x) for x in forward_pass_res["hidden_layer_output"]] 
                hidden_layer_error_gradient = [weight * output_loss * derv for weight, derv in zip(self.hidden_2_output, hidden_sigmoid_dervs)]  
                print(f"ouput_layer_gradient: {output_layer_error_gradient}")
                print(f"hidden_layer_gradient: {hidden_layer_error_gradient}")

                total_loss += output_loss
            print(f"Output_loss: {output_loss/self.epoch} for iteration {iter}")


if __name__ == "__main__":
    training_data = [[0, 0], [1, 1], [0, 1], [1, 0]]
    labels = [1, 1, 0, 0]
    p = Perceptron(epoch=2)
    p.train(training_data, labels)
    print(p.forward_pass([0, 0]))
    # print(neurons_mult(mat, vec, bias))
    # print(cross_entropy_loss(1000, 0))
