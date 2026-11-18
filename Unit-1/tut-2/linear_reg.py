# Tutorial 2: Simple Regression using Keras

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load online/standard dataset
data = load_diabetes()

# Use BMI as input feature
X = data.data[:, 2]

# Target value
y = data.target

# Convert X into 2D array
X = X.reshape(-1, 1)

# 2. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale the input data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create Keras model
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(1,))
])

# 5. Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 6. Train the model
model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=16,
    verbose=1
)

# 7. Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)

print("Test Loss:", loss)
print("Test MAE:", mae)

# 8. Make a prediction
prediction = model.predict(X_test[:1])

print("Actual value:", y_test[0])
print("Predicted value:", prediction[0][0])