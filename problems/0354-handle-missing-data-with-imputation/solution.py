import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    data = np.array(data, dtype=float)       # makes a copy, so the input isn't changed

    for j in range(data.shape[1]):
        col = data[:, j]
        missing = np.isnan(col)
        present = col[~missing]

        if present.size == 0:                # all NaN: leave the column unchanged
            continue

        if strategy == 'mean':
            fill = present.mean()
        elif strategy == 'median':
            fill = np.median(present)
        elif strategy == 'mode':
            vals, counts = np.unique(present, return_counts=True)
            fill = vals[np.argmax(counts)]
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        col[missing] = fill

    return data