# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.

    Strategy: closed-form ordinary least squares (no learning rate,
    no epochs, deterministic). We append a column of ones so the bias
    is learned together with W, then solve min ||A @ theta - y||^2
    with np.linalg.lstsq (SVD-based, numerically stable).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    # Design matrix with a bias column: A = [X | 1]
    A = np.hstack([X, np.ones((X.shape[0], 1))])

    theta, *_ = np.linalg.lstsq(A, y, rcond=None)

    W_out = theta[:-1]
    b_out = float(theta[-1])
    return W_out, b_out