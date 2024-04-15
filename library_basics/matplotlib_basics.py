import matplotlib.pyplot as plt

# Sample data: epochs and their corresponding loss values
epochs = [1, 2, 3, 4, 5]
loss = [0.25, 0.15, 0.10, 0.05, 0.01]

plt.figure(figsize=(10, 5))
plt.plot(epochs, loss, marker='o', color='b', label='Training Loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
