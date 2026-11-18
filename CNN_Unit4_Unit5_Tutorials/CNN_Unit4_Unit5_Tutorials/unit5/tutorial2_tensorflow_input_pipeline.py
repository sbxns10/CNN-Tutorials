"""
Unit 5 - Tutorial 2: Build TensorFlow Input Pipelines for Images
----------------------------------------------------------------
Objective:
Demonstrate how to build an efficient TensorFlow tf.data input pipeline for
the cats_dogs dataset (Cats, Dogs) using:
- tf.keras.utils.image_dataset_from_directory()
- Training and validation splitting
- Storing class_names BEFORE applying dataset transformations
- cache(), shuffle(), and prefetch(buffer_size=tf.data.AUTOTUNE)
- Printing class names and batch shapes
- Training and evaluating a simple CNN using the configured pipeline
"""

import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

# Set random seed for reproducibility
tf.random.set_seed(42)

# 1. Define dataset path using pathlib
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "cats_dogs"

# Check if dataset exists
if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.png")):
    print(f"Error: Cats and Dogs dataset not found at '{DATASET_DIR}'.")
    print("Please run 'python generate_datasets.py' first.")
    sys.exit(1)

# Hyperparameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 5

print("=" * 70)
print("UNIT 5 - TUTORIAL 2: Build TensorFlow Input Pipelines for Images")
print("=" * 70)
print(f"Loading Cats & Dogs dataset from: {DATASET_DIR}")

# 2. Load training and validation splits using image_dataset_from_directory()
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# 3. IMPORTANT: Store class names BEFORE applying cache/shuffle/prefetch transformations
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nStored Class Names: {class_names}")

# 4. Inspect and print batch shapes from the dataset pipeline
for image_batch, labels_batch in train_ds.take(1):
    print(f"Image Batch Shape : {image_batch.shape}")
    print(f"Labels Batch Shape: {labels_batch.shape}")

# 5. Configure high-performance tf.data input pipeline using cache, shuffle, and prefetch
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    train_ds
    .cache()
    .shuffle(buffer_size=100)
    .prefetch(buffer_size=AUTOTUNE)
)

val_ds = (
    val_ds
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

print("\nInput pipeline configured with cache(), shuffle(), and prefetch(tf.data.AUTOTUNE).")

# 6. Build a simple CNN model using the loaded dataset pipeline
model = models.Sequential([
    layers.Rescaling(1.0 / 255, input_shape=(128, 128, 3)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax"),
])

# 7. Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# 8. Train the CNN using the optimized tf.data input pipelines
print("\nTraining CNN using the tf.data input pipeline...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# 9. Display model evaluation results
val_loss, val_accuracy = model.evaluate(val_ds, verbose=0)
print("\n" + "-" * 50)
print(f"Pipeline Evaluation Loss     : {val_loss:.4f}")
print(f"Pipeline Evaluation Accuracy : {val_accuracy * 100:.2f}%")
print("-" * 50)
