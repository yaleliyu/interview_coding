
from generate_data import *
import numpy as np

def train_linear_regression(l1: float = 0.0, l2: float = 0.0) -> np.ndarray:
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
        penalty = 2 * l2 * weights + l1 * np.sign(weights)
        penalty[0] = 0
        gradient = x.T @ (y_hat - y) / batch_size + penalty
        weights -= lr*gradient

        if epoch % 25 == 0:
            data_loss = np.mean(np.square(y_hat - y))
            reg = l2 * np.sum(weights[1:]**2) + l1 * np.sum(np.abs(weights[1:]))
            print(f"step {epoch} data_loss: {data_loss:.4f} reg: {reg:.4f}")

    print(f"weights: {weights}")
    print(f"true weights: {true_weights}")
    return weights

def train_logistic_regression(l1: float = 0.0, l2: float = 0.0) -> np.ndarray:
    features, labels, true_weights = generate_classification_data()
    n_features = features.shape[1]
    n_samples = features.shape[0]

    data = np.hstack((np.ones((n_samples, 1)), features, labels.reshape(-1,1)))
    batch_size = 32
    n_epochs = 1000
    weights = np.ones(n_features+1)
    lr = 0.01

    for epoch in range(n_epochs):
        idx = np.random.choice(n_samples, batch_size, replace=False)
        batch = data[idx]
        x = batch[:, :-1]
        y = batch[:, -1]
        z = x @ weights
        y_hat = np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        penalty = 2 * l2 * weights + l1 * np.sign(weights)
        penalty[0] = 0
        gradient = x.T @ (y_hat - y) / batch_size + penalty
        weights -= lr*gradient

        if epoch % 25 == 0:
            y_hat_c = np.clip(y_hat, 1e-12, 1 - 1e-12)
            data_loss = -np.mean(y*np.log(y_hat_c) + (1-y)*np.log(1-y_hat_c))
            reg = l2 * np.sum(weights[1:]**2) + l1 * np.sum(np.abs(weights[1:]))
            print(f"step {epoch} data_loss: {data_loss:.4f} reg: {reg:.4f}")

    print(f"weights: {weights}")
    print(f"true weights: {true_weights}")

    return weights





if __name__ == "__main__":
    train_linear_regression()
    train_logistic_regression()