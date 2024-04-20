import matplotlib.pyplot as plt
import numpy as np

class LossViz:
    def __init__(self):
        plt.figure(figsize=(10, 5))
        plt.title("Epoch vs Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

    def show(self, loss_data):
        data = np.array(loss_data)
        epochs = data[:,0]
        loss = data[:,1]
        plt.plot(epochs, loss, marker='o', markersize=2, linestyle='-', color="b", label="Training Loss")
        plt.show()
