import numpy as np


class ConfusionMatrix:
    """ A confusion matrix for outputs against targets.

    Rows are for targets (Observations) and
    columns are for outputs (Predictions).
    """
    def __init__(self, outputs, targets):
        """Construct a confusion matrix for outputs against targets.

        Args:
            outputs (iterable): Classification outputs (prediction)
                                for two or more classes.
            targets (iterable): Classification targets (Observation)
                                for two or more classes.
        """

        self.classes = set(targets)
        classes_len = len(self.classes)
        self.conf_matrix = np.zeros((classes_len, classes_len))

        self.classes_indices = dict(
            [(c, i) for i, c in enumerate(self.classes)])

        for (output, target) in zip(outputs, targets):
            # Find the indices for classes
            i = self.classes_indices[target]
            j = self.classes_indices[output]
            # Update the confusion matrix based on
            # the comparision of target against output
            self.conf_matrix[i, j] += 1

    def accuracy(self):
        return np.trace(self.conf_matrix)/np.sum(self.conf_matrix)

    def precision(self):
        return self.conf_matrix[1, 1]/np.sum(self.conf_matrix[:, 1])

    def recall(self):
        return self.conf_matrix[1, 1]/np.sum(self.conf_matrix[1, :])

    def sensitivity(self):
        return self.recall()

    def specificity(self):
        return self.conf_matrix[0, 0]/np.sum(self.conf_matrix[0, :])
