def dice_statistics(n: int) -> tuple[float, float]:
    expected_value = (n + 1) / 2
    variance = (n**2 - 1) / 12
    return (round(expected_value, 4), round(variance, 4))