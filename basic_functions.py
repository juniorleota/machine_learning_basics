import math

# Squash all numbers to 0...1
def sigmoid(x):
    return 1/ (1 + math.exp(-x))


if __name__ == '__main__':
    sigmoid(1)
