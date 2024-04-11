'''
This class will explore using np to simplify alot of the calculations done for MLP.
'''
import numpy as np
import math

def sigmoid(x):
    pass

def sigmoid_d(x):
    pass

def cross_entropy_loss(predicted_ouput, expected_output):
    pass

def cross_entropy_loss_d(predicted_ouput, expected_output):
    pass

class MLP:
    def __init__(self, epochs=1000, learning_rate=0.1):
        self.epochs = epochs
        self.lr = learning_rate
        # row is hidden neuron
        # col is input
        self.w_input_to_hidden = np.array([[0.1, 0.2],[0.3, 0.4]])
        self.b_hidden = np.zeros(2)
        self.w_hidden_to_output = np.array([0.25, 0.45])
        self.b_output = np.zeros(1)
    
    def forw_pass(self):
        pass
                  
    def back_pass(self):
        pass

    def train(self, training_data, labels):
        pass


if __name__ == "__main__":
    mlp = MLP()