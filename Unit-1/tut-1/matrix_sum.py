## Write a TensorFlow program to read two 2-dimensional tensors (matrices) of the same shape and compute 
# their element-wise sum. Display both input matrices and the resultant matrix.

import tensorflow as tf

# Read number of rows and columns
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

# Read first matrix
print("\nEnter elements of first matrix row by row:")

matrix1 = []

for i in range(rows):
    row = list(map(float, input().split()))
    matrix1.append(row)

# Read second matrix
print("\nEnter elements of second matrix row by row:")

matrix2 = []

for i in range(rows):
    row = list(map(float, input().split()))
    matrix2.append(row)

# Convert matrices to TensorFlow tensors
tensor1 = tf.constant(matrix1)
tensor2 = tf.constant(matrix2)

# Element-wise addition
result = tf.add(tensor1, tensor2)

# Display results
print("\nFirst Matrix:")
print(tensor1)

print("\nSecond Matrix:")
print(tensor2)

print("\nResultant Matrix:")
print(result)