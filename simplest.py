# Define the activation function for the perceptron
def step_function(input_value):
    """
    Determines the output of the perceptron based on the input value.
    
    :param input_value: The calculated input to the activation function.
    :return: 1 if input_value is non-negative, 0 otherwise.
    """
    if input_value >= 0:
        return 1
    else:
        return 0

class Perceptron:
    """
    Represents a single neuron with binary output, implemented without NumPy.
    """
    def __init__(self, learning_rate=0.01, n_iterations=10):
        self.learning_rate = learning_rate  # How quickly the perceptron can adjust its weights
        self.n_iterations = n_iterations  # Number of times the perceptron will go through the training set
        self.weights = []  # Placeholder for the weights of the perceptron
        self.bias = 0  # The bias term, which allows adjustment of the threshold

    def fit(self, training_inputs, training_labels):
        """
        Trains the perceptron using the provided data.
        
        :param training_inputs: List of input vectors.
        :param training_labels: Corresponding list of expected output values.
        """
        n_samples = len(training_inputs)
        n_features = len(training_inputs[0])

        # Initialize the weights for each feature to 0
        self.weights = [0 for _ in range(n_features)]

        # Iteratively adjust weights based on the training data
        for _ in range(self.n_iterations):
            for idx, input_vector in enumerate(training_inputs):
                linear_output = sum(weight * feature for weight, feature in zip(self.weights, input_vector)) + self.bias
                prediction = step_function(linear_output)

                # Weight update rule
                error = training_labels[idx] - prediction
                weight_update = self.learning_rate * error
                self.weights = [weight + weight_update * feature for weight, feature in zip(self.weights, input_vector)]
                self.bias += weight_update

    def predict(self, input_vectors):
        """
        Predicts the output for each input vector in the list.
        
        :param input_vectors: List of input vectors to predict output for.
        :return: List of predicted outputs.
        """
        predictions = []
        for input_vector in input_vectors:
            linear_output = sum(weight * feature for weight, feature in zip(self.weights, input_vector)) + self.bias
            prediction = step_function(linear_output)
            predictions.append(prediction)
        return predictions

# Example usage
if __name__ == "__main__":
    # Sample data and labels for AND logic operation
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    perceptron = Perceptron(learning_rate=0.1, n_iterations=10)
    perceptron.fit(X, y)

    predictions = perceptron.predict(X)
    print("Predictions for AND logic gate:", predictions)
