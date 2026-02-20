import numpy as np
from model import My2LP
import prep_data

# TODO: Add matplotlib visualizer

epochs = 10
lr = 0.111
batch = 128

model = My2LP()

####################################

X_train = prep_data.load_img()
Y_train = prep_data.load_lbl()

for epoch in range(epochs):
    avg_loss = 0
    count = 0
    perm = np.random.permutation(X_train.shape[1])
    X_shuf = X_train[:, perm]
    Y_shuf = Y_train[:, perm]

    print()
    print(f"-------------- EPOCH {epoch} --------------")
    print()

    for i in range(0, X_train.shape[1], batch):
        X_i = X_shuf[:, i:i+batch]
        Y_i = Y_shuf[:, i:i+batch]

        model.forward(X_i)

        curr_loss = model.loss(Y_i)
        avg_loss += curr_loss
        count += 1

        if i % (50 * batch) == 0:
            num_right = model.performance(Y_i)
            num_wrong = batch - num_right
            b = i // batch
            print(f"Batch {b}: Correct - {num_right}/{batch}")
            print(f"           Wrong:  - {num_wrong}/{batch}")
            print(f"           Loss: {curr_loss:.4f}")

        grads = model.backprop(Y_i)
        model.update_params(grads, lr)
    
    print(f"AVERAGE LOSS: {(avg_loss / count):4f}")

print("Sucess")

model.save_weights("./data/mnist_weights.npz")
