import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    # Values of g and h at x
    g = np.polyval(g_coeffs, x)
    h = np.polyval(h_coeffs, x)

    # Values of the derivatives g' and h' at x
    dg = np.polyval(np.polyder(g_coeffs), x)
    dh = np.polyval(np.polyder(h_coeffs), x)

    # Quotient rule: (g'h - gh') / h²
    return float((dg * h - g * dh) / h**2)