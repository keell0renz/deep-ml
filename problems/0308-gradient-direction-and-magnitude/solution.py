import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""

	grad = np.array(gradient)
	
	magnitude = np.linalg.norm(grad)
	unit = grad / np.linalg.norm(grad)
	direction = -unit

	return {
		"magnitude": magnitude,
		"direction": np.nan_to_num(unit),
		"descent_direction": direction
	}

	pass