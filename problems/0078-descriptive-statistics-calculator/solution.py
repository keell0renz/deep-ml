import numpy as np

def descriptive_statistics(data: list | np.ndarray) -> dict:
    data = np.array(data)

    mean = np.mean(data)
    median = np.median(data)

    values, counts = np.unique(data, return_counts=True)
    mode = values[np.argmax(counts)]

    variance = np.var(data)          # population variance (divide by N) by default
    std_dev = np.sqrt(variance)

    p25 = np.percentile(data, 25)
    p50 = np.percentile(data, 50)
    p75 = np.percentile(data, 75)
    iqr = p75 - p25

    return {
        'mean': round(float(mean), 4),
        'median': round(float(median), 4),
        'mode': mode.item(),
        'variance': round(float(variance), 4),
        'standard_deviation': round(float(std_dev), 4),
        '25th_percentile': round(float(p25), 4),
        '50th_percentile': round(float(p50), 4),
        '75th_percentile': round(float(p75), 4),
        'interquartile_range': round(float(iqr), 4),
    }