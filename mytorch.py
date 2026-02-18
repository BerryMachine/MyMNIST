import numpy as np

def relu(Z):
    return np.maximum(0, Z)

def leaky_relu(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)

# def softmax(Z):
#     expz = np.exp(Z - np.max(Z))
#     return expz / expz.sum()

def softmax(Z):
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    expz = np.exp(shift_Z)
    return expz / np.sum(expz, axis=0, keepdims=True)