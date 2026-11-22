import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)          # ddof=0 is the population std, as required
    sigma[sigma == 0] = 1.0              # constant column: divide by 1, so you get x - mu

    return (X_test - mu) / sigma