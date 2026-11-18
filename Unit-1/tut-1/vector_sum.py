#Write a TensorFlow program to read 
#two vectors of the same dimension and compute their element-wise sum. Display both input vectors and the resultant vector.

import tensorflow as tf

# Read vector dimension
n = int(input("Enter the dimension of the vectors: "))

# Read first vector
print("Enter elements of first vector:")
vector1 = list(map(float, input().split()))

# Read second vector
print("Enter elements of second vector:")
vector2 = list(map(float, input().split()))

# Convert lists to TensorFlow tensors
tensor1 = tf.constant(vector1)
tensor2 = tf.constant(vector2)

# Element-wise addition
result = tf.add(tensor1, tensor2)

# Display results
print("\nFirst Vector:")
print(tensor1)

print("\nSecond Vector:")
print(tensor2)

print("\nResultant Vector:")
print(result)