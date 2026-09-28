def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	m, n = len(matrix), len(matrix[0])

	means = []

	if mode == "row":
		for m_i in range(m):
			means.append(sum(matrix[m_i]) / n)

	if mode == "column":
		for n_i in range(n):
			columns = [matrix[m_i][n_i] for m_i in range(m)]
			means.append(sum(columns) / m)

	return means