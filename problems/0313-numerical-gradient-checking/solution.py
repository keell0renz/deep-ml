import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    x = np.asarray(x, dtype=float)
    analytical_grad = np.asarray(analytical_grad, dtype=float)
    numerical_grad = np.zeros_like(x)

    for i in range(x.size):
        x_plus = x.copy();  x_plus.flat[i] += epsilon
        x_minus = x.copy(); x_minus.flat[i] -= epsilon
        numerical_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)

    denom = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
    if denom == 0:
        relative_error = 0.0
    else:
        relative_error = np.linalg.norm(numerical_grad - analytical_grad) / denom

    return numerical_grad, float(relative_error)