import numpy as np

def one_hot(Y, classes=10):
    v = np.zeros((classes, Y.size))
    v[Y, np.arange(Y.size)] = 1
    return v

def load_img(images_path="./data/train-images.idx3-ubyte"):
    with open(images_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint8, offset=16).astype(np.float32)

    data = data.reshape(-1, 784).T
    return data / 255.0 # data is (784, batch)

def load_lbl(labels_path="./data/train-labels.idx1-ubyte"):
    with open(labels_path, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8, offset=8)
    
    return one_hot(labels) # labels is (10, batch)