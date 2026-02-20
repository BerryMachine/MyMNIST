import numpy as np
from mytorch import (relu, softmax)

IN_SIZE = 784
H_SIZE = 128
OUT_SIZE = 10

# MODEL_PATH = "./data/model.pppp"

# X (784, batch)
# W1 (128, 784)
# b1 (128, 1)
# Z1 (128, batch)

# A1 (128, batch)
# W2 (10, 128)
# b2 (10, 1)
# Z2 (10, batch) - result

# A2 (10, batch) - softmax

class My2LP:
    def __init__(self, weight_path=None, in_size=IN_SIZE, h_size=H_SIZE, out_size=OUT_SIZE):
        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        self.params = {}
        self.cache = {}

        if weight_path:
            self.load_weights(weight_path)
        else:
            self.init_params()

    def load_weights(self, weight_path):
        weights = np.load(weight_path)
        self.params["W1"] = weights["W1"]
        self.params["b1"] = weights["b1"]
        self.params["W2"] = weights["W2"]
        self.params["b2"] = weights["b2"]

    def save_weights(self, weight_path):
        np.savez(weight_path, **self.params)

    def init_params(self):
        std1 = np.sqrt(2.0 / self.in_size).astype(np.float32)
        self.params["W1"] = (np.random.randn(self.h_size, self.in_size) * std1).astype(np.float32)
        self.params["b1"] = np.zeros((self.h_size, 1), dtype=np.float32)

        std2 = np.sqrt(2.0 / self.h_size).astype(np.float32)
        self.params["W2"] = (np.random.randn(self.out_size, self.h_size) * std2).astype(np.float32)
        self.params["b2"] = np.zeros((self.out_size, 1), dtype=np.float32)

    def forward(self, X: np.ndarray):
        self.cache["X"] = X
        self.cache["Z1"] = self.params["W1"] @ X + self.params["b1"]
        self.cache["A1"] = relu(self.cache["Z1"])
        self.cache["Z2"] = self.params["W2"] @ self.cache["A1"] + self.params["b2"]
        self.cache["A2"] = softmax(self.cache["Z2"])
    
    def backprop(self, Y: np.ndarray):
        batch = Y.shape[1]

        dZ2 = self.cache["A2"] - Y # cross-entropy + softmax

        dW2 = (dZ2 @ self.cache["A1"].T) / batch
        db2 = np.sum(dZ2, axis=1, keepdims=True) / batch
        dA1 = (self.params["W2"].T @ dZ2)

        dZ1 = dA1 * (self.cache["Z1"] > 0) # relu

        dW1 = (dZ1 @ self.cache["X"].T) / batch
        db1 = np.sum(dZ1, axis=1, keepdims=True) / batch

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    def loss(self, Y: np.ndarray):
        batch = Y.shape[1]
        log_p = np.log(self.cache["A2"] + 1e-8)
        return -np.sum(Y * log_p) / batch
    
    def performance(self, Y: np.ndarray):
        predictions = np.argmax(self.cache["A2"], axis=0)
        targets = np.argmax(Y, axis=0)
        return np.sum(predictions == targets)
    
    def update_params(self, grads, lr):
        self.params["W1"] -= grads["dW1"] * lr
        self.params["b1"] -= grads["db1"] * lr
        self.params["W2"] -= grads["dW2"] * lr
        self.params["b2"] -= grads["db2"] * lr

    def inference(self, X: np.ndarray):
        Z1 = self.params["W1"] @ X + self.params["b1"]
        A1 = relu(Z1)
        Z2 = self.params["W2"] @ A1 + self.params["b2"]
        A2 = softmax(Z2)
        return np.argmax(A2)