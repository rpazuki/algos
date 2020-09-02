import numpy as np


def entropy(distribution, unit=2):
    """Finds the entropy of a distribution.
        Its default unit is 'bit'.

    Args:
        distribution ([FrequencyTable]): A FrequencyTable or any of
                                         its sub-classes
        unit (int, optional): Unit of the entropy. Defaults to 2 (bits).

    Returns:
        [float]: Entropy of the distribution
    """
    frequencies = distribution.frequencies(normalised=True)
    # check to see if it is a deterministic case (all but one are zero)
    zeros_size = frequencies[frequencies == 0].size
    if zeros_size + 1 == frequencies.size:
        return 0
    else:
        return np.sum(-frequencies * np.log2(frequencies) / np.log2(unit))
