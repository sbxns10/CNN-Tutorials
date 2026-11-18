# tutorial1_conv_layers.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# For reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Training settings
EPOCHS = 1
BATCH_SIZE = 128


def load_mnist():
    """
    Loads MNIST dataset and reshapes images for CNN.
    Input shape becomes: (samples, 28, 28, 1)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for grayscale images
    # Shape: (samples, 28, 28) -> (samples, 28, 28, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_conv_model():
    """
    CNN model with convolution layers.
    """
    model = models.Sequential([
        # First convolution layer
        # Input: 28x28 grayscale image
        # Output: 26x26 feature maps with 32 filters
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1)
        ),

        # Second convolution layer
        # Output: 24x24 feature maps with 64 filters
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Convert 2D feature maps to 1D vector
        layers.Flatten(),

        # Output layer: 10 classes for digits 0-9
        layers.Dense(10, activation="softmax")
    ])

    return model


def main():
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    print("=" * 50)
    print("Tutorial 1: CNN with Convolution Layers")
    print("=" * 50)

    print("Train input shape:", x_train.shape)
    print("Test input shape: ", x_test.shape)
    print("Example image shape:", x_train[0].shape)
    print("Example label:", y_train[0])

    # Build model
    model = build_conv_model()

    # Show model structure
    model.summary()

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train model
    print("\nTraining started...")
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Predict first 5 test images
    sample_images = x_test[:5]
    true_labels = y_test[:5]

    probabilities = model.predict(sample_images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)

    print("\nSample prediction output")
    print("Model output shape:", probabilities.shape)
    print("Predicted labels:", predicted_labels)
    print("True labels:      ", true_labels)


if __name__ == "__main__":
    main()