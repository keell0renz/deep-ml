import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """

    def mse(X_b, w, y_b):
        return np.mean((X_b @ w - y_b) ** 2)

    def mse_grad(X_b, w, y_b):
        k = len(y_b)
        return (2 / k) * X_b.T @ (X_b @ w - y_b)

    def batch():
        if method == "batch":
            yield X, y
            return
        
        step = 1 if method == "stochastic" else batch_size

        for i in range(0, len(X), step):
            yield X[i:i + step], y[i:i + step]

    for epoch in range(n_epochs):
        for X_b, y_b in batch():
            g = mse_grad(X_b, weights, y_b)

            weights += -learning_rate * g

    return weights
