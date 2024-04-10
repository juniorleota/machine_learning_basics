import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# This function is only used for analysis, only its derivative is used for analysis
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
    return [x * scalar for x in vec]


def vector_sub(vec_a, vec_b):
    return [x - y for x, y in zip(vec_a, vec_b)]


def vector_mult(vector_a, vector_b):
    res = 0
    for x, y in zip(vector_a, vector_b):
        res += x * y
    return res


def mat_sub(mat_a, mat_b):
    new_mat = []
    for row_a, row_b in zip(mat_a, mat_b):
        new_row = []
        for col_a, col_b in zip(row_a, row_b):
            new_row.append(col_a - col_b)
        new_mat.append(new_row)
    return new_mat


def matrix_scale(mat, scalar):
    new_mat = []
    for row in mat:
        new_row = []
        for col in row:
            new_row.append(col * scalar)
        new_mat.append(new_row)
    return new_mat


def vect_mult_mat(v_col, v_row):
    res = []
    for col_elem in v_col:
        res_row = []
        for row_elem in v_row:
            res_row.append(col_elem * row_elem)
        res.append(res_row)
    return res


class Perceptron:
    def __init__(self, learning_rate=0.01, epoch=100):
        # todo use random generated values for weights for now we will use static values for ease of debugging
        self.input_2_hidden_weights = [[0.1, 0.2], [0.3, 0.4]]
        self.hidden_bias = [0, 0]
        self.hidden_2_output_weights = [0.25, 0.45]
        self.output_bias = 0
        self.learning_rate = learning_rate
        self.epoch = epoch

    def forward_pass(self, input_vector):
        hidden_layer_output = [
            vector_mult(neuron_weights, input_vector) + bias
            for neuron_weights, bias in zip(
                self.input_2_hidden_weights, self.hidden_bias
            )
        ]
        hidden_layer_activation = [sigmoid(output) for output in hidden_layer_output]
        # no need to transpose since there is only one output neuron
        output_layer_output = (
            vector_mult(self.hidden_2_output_weights, hidden_layer_activation)
            + self.output_bias
        )
        output_layer_activation = sigmoid(output_layer_output)
        res = {
            "hidden_layer_output": hidden_layer_output,
            "hidden_layer_activation": hidden_layer_activation,
            "output_layer_output": output_layer_output,
            "output_layer_activation": output_layer_activation,
        }

        # print("Forward pass result")
        # for key, val in res.items():
        #    print(f"> {key}:{val}")

        return res

    """
    This whole thing is a matrix chain rule implementation going backwards:
    1. derv ouput activation w.r.t to predicted ouput
    2. derv ouput z w.r.t to 
    """

    def backward_pass(self, input_vector, expected_ouput, forward_pass_res):
        output_layer_output = forward_pass_res["output_layer_output"]
        output_layer_activation = forward_pass_res["output_layer_activation"]
        hidden_layer_output = forward_pass_res["hidden_layer_output"]
        hidden_layer_activation = forward_pass_res["hidden_layer_activation"]

        # Derive output layer
        derv_output_activation = cross_entropy_loss_derivative(
            output_layer_activation, expected_ouput
        )
        derv_output_z = derv_output_activation * sigmoid_derivative(output_layer_output)
        # todo this needs to be a matrix so that it can be subtracted later
        derv_hidden_2_output_weights = [
            derv_output_z * hidden_neuron_activation
            for hidden_neuron_activation in hidden_layer_activation
        ]
        derv_output_bias = derv_output_z

        # derive hidden
        # we go backwards here
        # (hidden_bias | input_2_hiddden_weights | input_activation) -> hidden_ouput ->
        # (hidden_activation | hidden_2_ouput_weights | output_bias) -> ouput_z -> output_activation
        derv_hidden_activation = [
            neuron_weights * derv_output_z
            for neuron_weights in self.hidden_2_output_weights
        ]
        derv_hidden_output = [
            hidden_neuron_a * sigmoid_derivative(hidden_neuron_z)
            for hidden_neuron_z, hidden_neuron_a in zip(
                hidden_layer_output, derv_hidden_activation
            )
        ]
        derv_input_2_hidden_weights = vect_mult_mat(input_vector, derv_hidden_output)
        derv_hidden_bias = derv_hidden_output

        return (
            derv_hidden_2_output_weights,
            derv_output_bias,
            derv_input_2_hidden_weights,
            derv_hidden_bias,
        )

    def train(self, training_data, labels):
        for iter in range(self.epoch):
            total_loss = 0
            # print(f"Iteration {iter}")
            for input_vector, expected_output in zip(training_data, labels):
                forward_pass_res = self.forward_pass(input_vector)
                (
                    derv_hidden_2_output_weights,
                    derv_output_bias,
                    derv_input_2_hidden_weights,
                    derv_hidden_bias,
                ) = self.backward_pass(input_vector, expected_output, forward_pass_res)
                lr = self.learning_rate
                self.hidden_2_output_weights = vector_sub(
                    self.hidden_2_output_weights,
                    vector_scale(derv_hidden_2_output_weights, lr),
                )
                self.output_bias -= lr * derv_output_bias
                self.input_2_hidden_weights = mat_sub(
                    self.input_2_hidden_weights,
                    matrix_scale(derv_input_2_hidden_weights, lr),
                )
                self.hidden_bias = vector_sub(
                    self.hidden_bias, vector_scale(derv_hidden_bias, lr)
                )
                output_activation = forward_pass_res["output_layer_activation"]
                iter_loss = cross_entropy_loss(output_activation, expected_output)
                total_loss += iter_loss
            epoch_loss = total_loss / len(training_data)
            if iter % 100 == 0:
                print(f"Output_loss: {epoch_loss} for iter {iter}")


def full_training():
    training_data = [[0, 0], [1, 1], [0, 1], [1, 0]]
    labels = [0, 0, 1, 1]
    p = Perceptron(epoch=100000, learning_rate=0.1)
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
    full_training()
