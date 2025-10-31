import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.

# Task 1: 
# Instructions:
#Write a function that takes one numeric argument as input. 
#If the number is larger than zero, the function should return 1, otherwise is should return -1.
#The name of the function should be step

# Your code here:
# -----------------------------------------------

def step(num):
    if num > 0:
        return(1)
    else:
        return(-1)
    
step(5)
step(-5)


# -----------------------------------------------


# Task 2:
# Instructions:
#Write a function that takes in two arguments: a numpy array, and an integer (call argument "cutoff" and set default to 0).
#The function should return a numpy array of the same length, with all elements smaller than the cutoff being set to cutoff).
#The name of the function should be ReLu


# Your code here:
# -----------------------------------------------
def ReLu(array, cutoff = 0):
    array = np.array(array)
    return np.maximum(array, cutoff)

array = np.array([-5, 0, 3, -2, 4])

ReLu(array)

array = np.array([-5, 0, 3, -2, 4])
ReLu(array, 2)

array = np.array([])
ReLu(array)
# -----------------------------------------------


# Task 3:
# Instructions:
#Write a function that takes in a two-dimensional numpy array of size (n, p) and a one-dimensional numpy array of size p.
#The function should start by multiplying the two numpy arrays (matrix multiplication).
#Next, apply the ReLu function from above to the resulting matrix and return the result.
#Name the function neural_net_layer

# Your code here:
# -----------------------------------------------

def neural_net_layer(matrix, vector, cutoff = 0):
    if np.ndim(matrix) != 2 or np.ndim(vector) != 1:
        return("Error: matrix must have 2 dimensions. Vector must have 1 dimension")
    if np.shape(matrix[1]) != np.shape(vector):
        return("Error: matrix and vector shapes not compatible")
    matrix = np.asarray(matrix)
    vector = np.asarray(vector)
    new_matrix = matrix @ vector
    return ReLu(new_matrix, cutoff)


inputs = np.array([[1, 2], [3, 4]])
weights = np.array([1, -1])

neural_net_layer(inputs, weights)

inputs = np.array([[2, 3], [1, 1]])
weights = np.array([1, 2])

neural_net_layer(inputs, weights)

inputs = np.array([[1, -1], [-2, 3]])
weights = np.array([-2, 1])

neural_net_layer(inputs, weights)


# ------------------------------------------