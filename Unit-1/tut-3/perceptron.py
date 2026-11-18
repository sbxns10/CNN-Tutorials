# Tutorial 3: Perceptron using TensorFlow Keras

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Iris dataset
iris = load_iris()

# Take first two classes only
X = iris.data[:100]
y = iris.target[:100]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Standardize features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create perceptron model
model = keras.Sequential([
    keras.layers.Dense(
        1,
        activation='sigmoid',
        input_shape=(4,)
    )
])

# 5. Compile model
model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 6. Train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# 7. Evaluate model
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# 8. Make predictions
predictions = model.predict(X_test)

# Convert probabilities into classes
predicted_classes = (predictions >= 0.5).astype(int)

print("Actual classes:")
print(y_test)

print("Predicted classes:")
print(predicted_classes.flatten())