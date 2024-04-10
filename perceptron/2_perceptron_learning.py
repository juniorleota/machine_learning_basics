class Perceptron:
    def __init__(self, training_iterations, learning_rate, threshold=0):
        self.weights = [10, 10]
        self.bias = 10
        self.learning_rate = learning_rate
        self.training_iterations = training_iterations
        self.threshold = threshold

    def activation(self, x):
        if x >= self.threshold:
            return 1
        return 0

    def weighted_sum(self, a, b):
        res = 0
        for x, y in zip(a, b):
            res += x * y
        return res

    def predict(self, feature):
        linear_output = self.weighted_sum(feature, self.weights) + self.bias
        return self.activation(linear_output)

    def train(self, features, labels):
        for epoch in range(self.training_iterations):
            # input vector has 2 values
            all_pass = True
            for input_vector, y_expected in zip(features, labels):
                y_prediction = self.predict(input_vector)
                new_weights = []
                if y_prediction != y_expected:
                    all_pass = False
                for x, w_old in zip(input_vector, self.weights):
                    w_new = w_old + self.learning_rate * (y_expected - y_prediction) * x
                    new_weights.append(w_new)
                new_bias = self.bias + self.learning_rate * (y_expected - y_prediction)
                self.weights = new_weights
                self.bias = new_bias
            print(f"Epoch {epoch}: weights({self.weights}), bias({self.bias})")
            if all_pass:
                print("All training data passed")
                break


if __name__ == "__main__":
    training_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    label = [0, 0, 0, 1]
    perceptron = Perceptron(100, 0.09, 0)
    perceptron.train(training_data, label)
    for input in training_data:
        output = perceptron.predict(input)
        print(f"input({input}) => ouput({output})")
