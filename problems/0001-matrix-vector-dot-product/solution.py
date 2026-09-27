def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Every row must have as many columns as b has entries
    for row in a:
        if len(row) != len(b):
            return -1

    result = []
    for row in a:
        total = 0
        for x, y in zip(row, b):
            total += x * y
        result.append(total)
    return result