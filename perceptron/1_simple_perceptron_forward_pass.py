import math

def sigmoid(x):
    return 1/(1+math.exp(-x))

def activation(x):
    if x>0.5:
        return 1
    return 0

def prediction(feature, weights, bias):
    weighted_sum = 0
    for x,weight in zip(feature, weights):
        weighted_sum += x*weight
    
    linear_output = weighted_sum + bias
    print(f"linear_output = {linear_output}")
    return activation(linear_output)
if __name__ == '__main__':
    feature = [0.1, 0.1, 0.1]
    weights = [0.1, 0.5, 0.9]
    bias = 0.3
    prediction(feature, weights, bias)