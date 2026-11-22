# --- cell 1 ---
import numpy as np

def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # shuffle, then cut into three disjoint folds
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]        # the remainder, so every row is used once

    # least squares with an intercept, fitted on training rows only
    A_train = np.column_stack([X[train_idx], np.ones(len(train_idx))])
    w, *_ = np.linalg.lstsq(A_train, y[train_idx], rcond=None)

    A_test = np.column_stack([X[test_idx], np.ones(len(test_idx))])
    test_predictions = A_test @ w

    return test_predictions, train_idx, val_idx, test_idx