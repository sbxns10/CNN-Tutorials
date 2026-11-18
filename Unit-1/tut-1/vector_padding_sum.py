## Write a TensorFlow program to add two vectors of different dimensions by padding the shorter vector with 
# zeros so that both vectors have equal length before performing the addition. Display the padded vector and the resultant sum.

import tensorflow as tf

# Read first vector
print("Enter elements of first vector:")
vector1 = list(map(float, input().split()))

# Read second vector
print("Enter elements of second vector:")
vector2 = list(map(float, input().split()))

# Convert to TensorFlow tensors
tensor1 = tf.constant(vector1)
tensor2 = tf.constant(vector2)

# Find the maximum dimension
max_length = max(len(vector1), len(vector2))

# Calculate required padding
padding1 = max_length - len(vector1)
padding2 = max_length - len(vector2)

# Pad the shorter vector with zeros
padded_tensor1 = tf.pad(tensor1, [[0, padding1]])
padded_tensor2 = tf.pad(tensor2, [[0, padding2]])

# Perform element-wise addition
result = tf.add(padded_tensor1, padded_tensor2)

# Display results
print("\nFirst Vector:")
print(tensor1)

print("\nSecond Vector:")
print(tensor2)

print("\nPadded First Vector:")
print(padded_tensor1)

print("\nPadded Second Vector:")
print(padded_tensor2)

print("\nResultant Vector:")
print(result)