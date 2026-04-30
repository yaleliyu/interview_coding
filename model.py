
from generate_data import *
import numpy as np

def train_linear_regression() -> np.ndarray:
    features, labels, true_weights = generate_regression_data()
    n_features = features.shape[1]
    n_samples = features.shape[0]
    data = np.hstack((np.ones((n_samples, 1)), features, labels.reshape(-1,1)))

    batch_size = 32
    n_epochs = 1000
    weights = np.ones(n_features+1)
    lr = 0.02

    for epoch in range(n_epochs):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        batch = data[idx]
        x = batch[:, :-1]
        y = batch[:, -1]
        y_hat = x @ weights
        gradient = x.T @ (y_hat - y) / batch_size
        weights -= lr*gradient

        if epoch % 25 == 0:
            loss = np.mean(np.square(y_hat - y))
            print(f"step {epoch} loss: {loss}")

    print(f"weights: {weights}")
    print(f"true weights: {true_weights}")
    return weights






if __name__ == "__main__":
    train_linear_regression()
