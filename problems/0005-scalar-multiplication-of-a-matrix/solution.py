def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	
	new = []

	for row in matrix:
		new_row = []
		for v in row:
			new_row.append(v * scalar)
		
		new.append(new_row)

	return new