import numpy as np


class ClassificationMatrix:
    """Provides measures of performance for prediction against targets.

        Members:
        self.classSymbol:
        self.classes: Unique list of objects that is found in 'targets'
                      and  prediction'. It is lexicographically ordered
                      like Python set convention.
        self.classes_lookup: A lookup of class symbols indices in
                                    conf_matrix.
        self.conf_matrix: Confusion matrix.


    Rows are for targets (Observations) and
    columns are for prediction (Predictions).
    """

    def __init__(self, predictions, targets, class_symbol=None):
        """Construct a classification matrix for prediction against targets.

        Args:
         predictions (iterable): Classification predictions
                                for two or more classes.
            targets (iterable): Classification targets (Observation)
                                for two or more classes.
            classSymbol (object):The symbol that takes as true to find
                                true-positives in case of binary
                                classification.
                                Whenever it is 'None', the first item of
                                targets takes as the symbol.

        """
        # Empty target or prediction
        if len(targets) == 0:
            raise ValueError("'targets' cannot be empty.")
        if len(predictions) == 0:
            raise ValueError("'predictions' cannot be empty.")
        # In case, store the first element of the targets
        # to use as the symbol for the true cases, anytime
        # that the caller does not provide it
        self.class_symbol = class_symbol
        if self.class_symbol is None:
            self.class_symbol = targets[0]
        # Classes symbols
        self.classes = set(targets).union(set(predictions))
        # The confusion matrix for calculations
        self.conf_matrix = np.zeros((len(self.classes), len(self.classes)))
        # create a lookup for the position of symbols in 'conf_matrix'
        self.classes_lookup = dict([(c, i) for i, c in enumerate(self.classes)])

        for (output, target) in zip(predictions, targets):
            # Find the indices for classes from the lookup
            i = self.classes_lookup[target]
            j = self.classes_lookup[output]
            # Update the confusion matrix based on
            # the comparision of target against output
            self.conf_matrix[i, j] += 1

    def _get_class_symbole_index(self, class_symbol):
        if class_symbol is None:
            return self.classes_lookup[self.class_symbol]
        else:
            try:
                return self.classes_lookup[class_symbol]
            except KeyError:
                raise KeyError(
                    f"The class '{class_symbol}' is not defined "
                    + "in predictions or targets."
                )

    def tp(self, class_symbol=None):
        """True Positives

            For Binary classification
               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            Int: True Positives
        """
        i = self._get_class_symbole_index(class_symbol)
        return self.conf_matrix[i, i]

    def fp(self, class_symbol=None):
        """False Positives

            For Binary classification
               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            Int: False Positives
        """
        j = self._get_class_symbole_index(class_symbol)
        # Select the column of the class
        column = self.conf_matrix[:, j]
        return np.sum([v for k, v in enumerate(column) if k != j])

    def fn(self, class_symbol=None):
        """False Negative

            For Binary classification
               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            Int: False Negative
        """
        i = self._get_class_symbole_index(class_symbol)
        # Select the row of the class
        row = self.conf_matrix[i, :]
        return np.sum([v for k, v in enumerate(row) if k != i])

    def tn(self, class_symbol=None):
        """True Negative

            For Binary classification
               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            Int: True Negative
        """
        i = self._get_class_symbole_index(class_symbol)
        # The diagonal of the 'conf_matrix', except the Tap
        # is TN
        diagonals = np.diag(self.conf_matrix)
        return sum([v for k, v in enumerate(diagonals) if k != i])

    def accuracy(self):
        """Finds the accuracy of the correctly classified cases.

            It is equal to the ratio of the trace divided by total sum.
            For 'Binray Classification', it is
            (#TP + #TN)/(#TP + #FP + #TN + #FN)


        Returns:
            float: accuracy of correctly classified cases.

               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        """
        return np.trace(self.conf_matrix) / np.sum(self.conf_matrix)

    def precision(self, class_symbol=None):
        """Finds the precision of the correctly classified cases.

            For 'Binray Classification', it is
            #TP /(#TP + #FP)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: precision of correctly classified cases.

               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |

        """
        true_positives = self.tp(class_symbol)
        false_positives = self.fp(class_symbol)
        if true_positives + false_positives == 0:
            return 0
        else:
            return true_positives / (true_positives + false_positives)

    def recall(self, class_symbol=None):
        """Finds the recall of the correctly classified cases.

            For 'Binray Classification', it is
            #TP /(#TP + #FN)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: recall of correctly classified cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        true_positives = self.tp(class_symbol)
        false_positives = self.fn(class_symbol)
        if true_positives + false_positives == 0:
            return 0
        else:
            return true_positives / (true_positives + false_positives)

    def sensitivity(self, class_symbol=None):
        """Finds the sensitivity of the correctly classified cases.

            For 'Binray Classification', it is
            #TP /(#TP + #FN)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: sensitivity of correctly classified cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        true_positives = self.tp(class_symbol)
        false_positives = self.fn(class_symbol)
        if true_positives + false_positives == 0:
            return 0
        else:
            return true_positives / (true_positives + false_positives)

    def specificity(self, class_symbol=None):
        """Finds the specificity of the correctly classified cases.

            For 'Binray Classification', it is
            #TN /(#TN + #FP)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: specificity of correctly classified cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        true_negatives = self.tn(class_symbol)
        false_positives = self.fp(class_symbol)
        if true_negatives + false_positives == 0:
            return 0
        else:
            return true_negatives / (true_negatives + false_positives)

    def f1(self, class_symbol=None):
        """F1 score of correctly classified cases.

            f1 = (2 * precision * recall)/ (precision + recall)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: f1 score of correctly classified cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        return self.f_beta(beta=1, class_symbol=class_symbol)

    def matthews_corrcoef(self, class_symbol=None):
        """Matthews correlation coefficient of correctly classified cases.

           MCC = (TP*TN - FP*FN)/ (TP+FP)(TP+FN)(TN+FP)(TN+FN)

        Args:
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: Matthews correlation coefficient of correctly classified
                   cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        true_positives = self.tp(class_symbol)
        true_negatives = self.tn(class_symbol)
        false_positives = self.fp(class_symbol)
        false_negatives = self.fn(class_symbol)
        denom = (
            (true_positives + false_positives)
            * (true_positives + false_negatives)
            * (true_negatives + false_positives)
            * (true_negatives + false_negatives)
        )
        if denom == 0:
            return np.inf
        else:
            return (
                true_positives * true_negatives - false_positives * false_negatives
            ) / denom

    def f_beta(self, beta=1.0, class_symbol=None):
        """F-beta score of correctly classified cases.

            f-beta =
            ((1+beta^2) * precision * recall)/ (beta^2 * precision + recall)

        Args:
            beta (float, optional): Weight factor to control the precision
                       or recall importance. The higher beta means more recall
                       importance. Defaults to 1.0.
            class_symbol (object, optional): the object that is taken as
                       true positive.
                       Defaults to None and take the first element of the
                       'targets' as true or the one that is specified in
                       constructor.

        Returns:
            float: f-beta score of correctly classified cases.


               |     Prediction
               |  0     |   1
            ------------------------
            t  |        |
            a 0|  TP    |   FN
            r-----------------------
            g  |        |
            e 1|  FP    |   TN
            t  |        |
        """
        precision = self.precision(class_symbol)
        recall = self.recall(class_symbol)
        beta_2 = beta ** 2
        if precision + recall == 0:
            return 0
        else:
            return ((1 + beta_2) * precision * recall) / (beta_2 * precision + recall)
