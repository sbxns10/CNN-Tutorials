"""
Unit 5 - Tutorial 3: Implement a CNN for Image Processing Using L2 Regularization
---------------------------------------------------------------------------------
Objective:
Implement a beginner-friendly Convolutional Neural Network (CNN) with L2 weight
regularization (tf.keras.regularizers.l2(0.001)) applied to both Conv2D and
Dense layer kernels on the general_images dataset (Apple, Banana, Orange).

Key Components:
- Conv2D layers with kernel_regularizer=tf.keras.regularizers.l2(0.001)
- MaxPooling2D layers
- Dense layers with kernel_regularizer=tf.keras.regularizers.l2(0.001)
- Adam optimizer
- Sparse categorical crossentropy loss and accuracy metric
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

# Set random seed for reproducibility
tf.random.set_seed(42)

# 1. Define dataset path using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "general_images"

# Check if dataset exists
if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.png")):
    print(f"Error: General images dataset not found at '{DATASET_DIR}'.")
    print("Please run 'python generate_datasets.py' first.")
    sys.exit(1)

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5
L2_FACTOR = 0.001

print("=" * 70)
print("UNIT 5 - TUTORIAL 3: CNN with L2 Regularization (l2 = 0.001)")
print("=" * 70)
print(f"Loading dataset from: {DATASET_DIR}")

# 2. Load training and test splits
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes ({num_classes}): {class_names}")

# 3. Define L2 regularizer with factor 0.001
l2_reg = regularizers.l2(L2_FACTOR)

# 4. Build CNN model with L2 regularization applied to Conv2D and Dense kernels
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(128, 128, 3)),

    layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_regularizer=l2_reg,
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_regularizer=l2_reg,
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_regularizer=l2_reg,
    ),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=l2_reg,
    ),
    layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=l2_reg,
    ),
])

# 5. Compile the model using Adam optimizer, sparse categorical crossentropy, and accuracy
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 6. Train the model
print("\nStarting training with L2 regularization...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
)

# 7. Evaluate and display test loss and test accuracy
test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
print("\n" + "-" * 50)
print(f"L2 Regularized CNN Test Loss     : {test_loss:.4f}")
print(f"L2 Regularized CNN Test Accuracy : {test_accuracy * 100:.2f}%")
print("-" * 50)
