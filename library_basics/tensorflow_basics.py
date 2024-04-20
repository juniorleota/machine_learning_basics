from tensorflow.keras.datasets import mnist

def mnist_load():
    print(mnist)
    # Load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Check the shape of the loaded data
    print("Training data shape:", train_images.shape)  # Should show (60000, 28, 28)
    print("Test data shape:", test_images.shape)       # Should show (10000, 28, 28)

    # Example of using the data
    print("First training image:", train_images[0])
    print("First training image type:", type(train_images[0]))
    print("First training image label:", train_labels[0])

if __name__ == "__main__":
    mnist_load()
