def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    deduplicated = set(samples)

    pmf = []

    for number in deduplicated:
        pmf.append((number, samples.count(number) / len(samples)))

    return pmf