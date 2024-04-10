import math

"""
Lets introduce the concepts of:
- sigmoid as activation function as we cannot have binary output but we need a constrained result between 0...1
- matrixes for multiple neurons and layers
- how to deal with transpose with arrays (although not in the most scaleable of ways should use zip(*arr) or np.array.T)
"""


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def dot_sum(vector_a, vector_b):
    sum = 0
    for x, y in zip(vector_a, vector_b):
        sum += x * y
    return sum


def forward_pass_neuron(input_vector, neuron_weights, layer_bias):
    return dot_sum(neuron_weights, input_vector) + layer_bias


class MultiLayerPerceptron:
    """
    Roughtly we will have the input which it 2 numbers (but we can treat it as the input layer)
    Each input number will connect to all neurons in hidden layer
    Each hidden layer neuron will ouptut an activated number and will pass that number to 1 ouput neuron
    TODO:
    - this has issues with neuron size flexibility, it needs to be able to add more feature size and also add more layers as needed
    """

    def __init__(self, learning_rate=0.1, epocs=100):
        # Each neuron has 2 weights from the input
        # idx 1 contains training vector 1 ( [0][0] is first number and [0][1] is sencond number for XOR operation)
        # so num 1 will connect to neuron 1 and 2, and num 2 will also have same connection
        self.w_input_to_hidden = [[1, 1], [1, 1]]
        # so there is 1 bias per neuron in hidden layer
        self.bias_hidden = [0, 0]
        # there is only one output neuron, so each of this weights is to map both hidden layer neurons
        self.w_hidden_to_output = [0, 0]
        self.bias_output = 0
        self.learning_rate = learning_rate
        self.epocs = epocs

    def forward_pass(self, input_vector):
        hidden_layer_activation = []
        for neuron_weights, neuron_bias in zip(
            self.w_input_to_hidden, self.bias_hidden
        ):
            hidden_neuron_output = forward_pass_neuron(
                neuron_weights, input_vector, neuron_bias
            )
            hidden_neuron_activation = sigmoid(hidden_neuron_output)
            hidden_layer_activation.append(hidden_neuron_output)
            hidden_layer_activation.append(hidden_neuron_activation)
            print(f"Hidden neuron output: {hidden_neuron_output}")
        output_neuron_output = forward_pass_neuron(
            self.w_hidden_to_output, input_vector, self.bias_output
        )
        print(f"Output neuron output: {output_neuron_output}")
        output_neuron_activation = sigmoid(output_neuron_output)
        return output_neuron_activation


if __name__ == "__main__":
    perceptron = MultiLayerPerceptron()
    inputs = [1, 1]
    res = perceptron.forward_pass(inputs)
    print(f"Ouput: {res}")
