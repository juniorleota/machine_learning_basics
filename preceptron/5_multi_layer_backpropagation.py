import math

"""
So this implementation doesn't work but I left it here to hightligh how initializing everything to zero will lead to symmetry issue
"""


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
    return -1 * (
        expected_output * math.log(predicted_output_clipped)
        + (1 - expected_output) * math.log(1 - predicted_output_clipped)
    )


def cross_entropy_loss_derivative(y_predicted, y_expected):
    return -(y_expected / y_predicted) + ((1 - y_expected) / (1 - y_predicted))


def vector_add(vector_a, vector_b):
    return [x + y for x, y in zip(vector_a, vector_b)]

def vector_scale(vec, scalar):
    return [x*scalar for x in vec]

def vector_sub(vec_a, vec_b):
    return [x - y for x,y in zip(vec_a, vec_b)]

def vector_mult(vector_a, vector_b):
    res = 0
    for x, y in zip(vector_a, vector_b):
        res += x * y
    return res


# each entry in weights is the connection to multiple neurons
def neuron_mat_mult(weights_matrix, input_vector, layer_biases):
    trans_mat = transpose_matrix(weights_matrix)
    return [
        neuron_vector_mult(weight, input_vector, bias)
        for weight, bias in zip(trans_mat, layer_biases)
    ]


def neuron_vector_mult(weights, input_vector, bias):
    return vector_mult(weights, input_vector) + bias


class Perceptron:
    def __init__(self, learning_rate=0.01, epoch=100):
        self.input_2_hidden_w = [[0.1, 0.2], [0.3, 0.4]]
        self.hidden_bias = [0, 0]
        self.hidden_2_output_w = [0.25, 0.45]
        self.output_bias = 0
        self.learning_rate = learning_rate
        self.epoch = epoch

    def forward_pass(self, input_vector):
        hidden_layer_output = [
            vector_mult(neuron_weights, input_vector) + bias
            for neuron_weights, bias in zip(self.input_2_hidden_w, self.hidden_bias)
        ]
        hidden_layer_activation = [sigmoid(output) for output in hidden_layer_output]
        # no need to transpose since there is only one output neuron
        output_layer_output = (
            vector_mult(self.hidden_2_output_w, hidden_layer_activation)
            + self.output_bias
        )
        output_layer_activation = sigmoid(output_layer_output)
        res = {
            "hidden_layer_output": hidden_layer_output,
            "hidden_layer_activation": hidden_layer_activation,
            "output_layer_output": output_layer_output,
            "output_layer_activation": output_layer_activation,
        }

        print("Forward pass result")
        for key, val in res.items():
            print(f"> {key}:{val}")

        return res

    """
    This whole thing is a matrix chain rule implementation going backwards:
    1. derv ouput activation w.r.t to predicted ouput
    2. derv ouput z w.r.t to 
    """

    def backward_pass(self, input_vector, expected_ouput, forward_pass_res):
        output_layer_z = forward_pass_res["output_layer_output"]
        output_activation = forward_pass_res["output_layer_activation"]
        hidden_layer_output = forward_pass_res["hidden_layer_output"]
        hidden_layer_activation = forward_pass_res["hidden_layer_activation"]

        # Derive output layer
        derv_output_activation = cross_entropy_loss_derivative(
            output_activation, expected_ouput
        )
        derv_output_z = derv_output_activation * sigmoid_derivative(output_layer_z)
        # todo this needs to be a matrix so that it can be subtracted later
        derv_hidden_2_output_w = [
            derv_output_z * hidden_neuron_activation
            for hidden_neuron_activation in hidden_layer_activation
        ]
        derv_output_bias = derv_output_z

        # derive hidden
        # we go backwards here
        # (hidden_bias | input_2_hiddden_w | input_activation) -> hidden_ouput ->
        # (hidden_activation | hidden_2_ouput_w | output_bias) -> ouput_z -> output_activation
        derv_hidden_activation = [
            neuron_w * derv_output_z for neuron_w in self.hidden_2_output_w
        ]
        derv_hidden_output = [
            hidden_neuron_a * sigmoid_derivative(hidden_neuron_z)
            for hidden_neuron_z, hidden_neuron_a in zip(
                hidden_layer_output, derv_hidden_activation
            )
        ]
        derv_input_2_hidden_w = [
            input * neuron_derv_z
            for input, neuron_derv_z in zip(input_vector, derv_hidden_output)
        ]
        derv_hidden_bias = derv_hidden_output

        return (
            derv_hidden_2_output_w,
            derv_output_bias,
            derv_input_2_hidden_w,
            derv_hidden_bias,
        )

    def train(self, training_data, labels):
        for iter in range(self.epoch):
            total_loss = 0
            print(f"Iteration {iter}")
            for input_vector, expected_output in zip(training_data, labels):
                forward_pass_res = self.forward_pass(input_vector)
                (
                    derv_hidden_2_output_w,
                    derv_output_bias,
                    derv_input_2_hidden_w,
                    derv_hidden_bias,
                ) = self.backward_pass(input_vector, expected_output, forward_pass_res)
                lr = self.learning_rate
                self.hidden_2_output_w = vector_sub(self.hidden_2_output_w, vector_scale(derv_hidden_2_output_w, lr))
                self.output_bias -= lr * derv_output_bias
                # self.input_2_hidden_w = []
                print(f"di2hw: {derv_input_2_hidden_w}")
                self.hidden_bias = vector_sub(self.hidden_bias, vector_scale(derv_hidden_bias, lr))

                output_activation = forward_pass_res["output_layer_activation"]
                total_loss += cross_entropy_loss(output_activation, expected_output)

            epoch_loss = total_loss / len(training_data)
            print(f"Output_loss: {epoch_loss} ")
            print(
                f"updated values:\nhidden_weights: {self.input_2_hidden_w}\nhidden_bias: {self.hidden_bias}\noutput_weights: {self.hidden_2_output_w}\noutput_bias: {self.output_bias}\n"
            )

def full_training():
    training_data = [[0, 0], [1, 1], [0, 1], [1, 0]]
    labels = [0, 0, 1, 1]
    p = Perceptron(epoch=1000, learning_rate=0.1)
    p.train(training_data, labels)
    for input in training_data:
        output = p.forward_pass(input)["output_layer_activation"]
        print(f"{input} XOR = {output}")


def single_iter_training():
    training_data = [[1, 0]]
    labels = [1]
    p = Perceptron(epoch=1, learning_rate=0.1)
    p.train(training_data, labels)


if __name__ == "__main__":
    single_iter_training()
    # print(neurons_mult(mat, vec, bias))
    # print(cross_entropy_loss(1000, 0))
