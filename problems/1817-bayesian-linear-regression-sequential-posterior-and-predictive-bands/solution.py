import numpy as np

def posterior(X, y, m0, S0, sigma2):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    m0 = np.asarray(m0, dtype=float).ravel()
    S0 = np.asarray(S0, dtype=float)
    P0 = np.linalg.inv(S0)
    SN = np.linalg.inv(P0 + X.T @ X / sigma2)
    SN = (SN + SN.T) / 2  # keep symmetric
    mN = SN @ (P0 @ m0 + X.T @ y / sigma2)
    return mN, SN

def sequential_posterior(X, y, m0, S0, sigma2):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    m = np.asarray(m0, dtype=float).ravel()
    S = np.asarray(S0, dtype=float)
    means = np.zeros((X.shape[0], X.shape[1]))
    for n in range(X.shape[0]):
        m, S = posterior(X[n:n+1], y[n:n+1], m, S, sigma2)
        means[n] = m
    return m, S, means

def predictive(X_test, mN, SN, sigma2):
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    mean = X_test @ mN
    var = sigma2 + np.einsum('ij,jk,ik->i', X_test, SN, X_test)
    return mean, var