def matrixmul(a: list[list[int|float]],
              b: list[list[int|float]]) -> list[list[int|float]]:
    # Inner dimensions must match: columns of a == rows of b
    if not a or not b or len(a[0]) != len(b):
        return -1

    m, n, p = len(a), len(b), len(b[0])
    c = []
    for i in range(m):            # each row of a
        row = []
        for j in range(p):        # each column of b
            total = 0
            for k in range(n):    # walk along the shared dimension
                total += a[i][k] * b[k][j]
            row.append(total)
        c.append(row)
    return c