from tensorflow.keras.datasets import mnist  # type: ignore


def mnist_load():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print("Training data shape:", train_images.shape)
    print("Test data shape:", test_images.shape)
    print("First training image:", train_images[0])
    print("First training flattened:", train_images[0].flatten())
    print("First training image type:", type(train_images[0]))
    print("First training image label:", train_labels[0])


if __name__ == "__main__":
    mnist_load()
