def poly_term_derivative(c: float, x: float, n: float) -> float:
    
    new_exponent = n - 1
    new_coefficient = c * n

    return new_coefficient * (x ** new_exponent)