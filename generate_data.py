import numpy as np



def generate_regression_data(n_samples : int = 1000, n_features: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.random.uniform(-1, 1, n_features+1)
    features = np.random.uniform(-1, 1, (n_samples, n_features))
    labels = np.hstack((np.ones((n_samples, 1)), features)) @ weights + np.random.normal(0, 0.3, n_samples)

    return features, labels, weights



def generate_classification_data(n_samples : int = 1000, n_features: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.random.uniform(-1, 1, n_features+1)
    features = np.random.uniform(-1, 1, (n_samples, n_features))

    y = np.hstack((np.ones((n_samples, 1)), features))  @ weights
    z = (1 / (1 + np.exp(-y))) + np.random.normal(0, 0.3, n_samples)
    labels = np.where(z > 0.5, 1, 0)

    return features, labels, weights


if __name__ == "__main__":
    features, labels, weights = generate_regression_data()

    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}, Weights shape: {weights.shape}")
    print(f"labels sample: {labels[:10]}, labe dtype: {labels.dtype}")