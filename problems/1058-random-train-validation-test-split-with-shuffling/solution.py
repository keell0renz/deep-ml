import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float,
                 seed: int = 123) -> list:
    n = data.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    shuffled = data[idx]

    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    return [shuffled[:train_end],
            shuffled[train_end:validation_end],
            shuffled[validation_end:]]