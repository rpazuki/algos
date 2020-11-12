import numpy as np


def entropy(table, unit=2):
    """Finds the entropy of a distribution.
        Its default unit is 'bit'.

    Args:
        table ([Table]): A table or any of its sub-classes
        unit (int, optional): Unit of the entropy. Defaults to 2 (bits).

    Returns:
        [float]: Entropy of the table
    """
    values = [v for v in table.values()]
    total = sum(values)
    normalised_values = [v / total for v in values]
    return np.sum([-v * np.log2(v + 1e-300) for v in normalised_values]) / np.log2(unit)
