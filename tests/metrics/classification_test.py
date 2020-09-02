import pytest
import numpy as np
from metrics.classification import ClassificationMatrix


def test_classification_matrix_two_classes_first_symbol():
    # 12 cases for two classes
    # eight ones and 4 zeros
    targets = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    #
    predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    #
    matrix = ClassificationMatrix(predictions, targets)
    tp_symbol = 1
    tn_symbol = 0
    i = matrix.classes_lookup[tp_symbol]
    j = matrix.classes_lookup[tn_symbol]
    assert matrix.conf_matrix[i, i] == 6
    assert matrix.conf_matrix[j, j] == 3

    assert matrix.accuracy() == 9 / 12
    assert matrix.precision() == 6 / 7
    assert matrix.recall() == 6 / 8
    assert matrix.sensitivity() == 6 / 8
    assert matrix.specificity() == 3 / 4


def test_classification_matrix_mismatch_in_class_names():
    targets = ["A"] * 2 + ["B"] * 3
    predictions = ["C"] * 2 + ["D"] * 3
    matrix = ClassificationMatrix(targets, predictions)
    assert matrix.tp("A") == 0
    assert matrix.tn("A") == 0
    assert matrix.fp("A") == 2
    assert matrix.fn("A") == 0

    assert matrix.tp("B") == 0
    assert matrix.tn("B") == 0
    assert matrix.fp("B") == 3
    assert matrix.fn("B") == 0

    assert matrix.tp("C") == 0
    assert matrix.tn("C") == 0
    assert matrix.fp("C") == 0
    assert matrix.fn("C") == 2

    assert matrix.tp("D") == 0
    assert matrix.tn("D") == 0
    assert matrix.fp("D") == 0
    assert matrix.fn("D") == 3

    assert matrix.accuracy() == 0

    assert matrix.precision("A") == 0
    assert matrix.recall("A") == 0
    assert matrix.sensitivity("A") == 0
    assert matrix.specificity("A") == 0
    assert matrix.f1("A") == 0
    assert matrix.matthews_corrcoef("A") == np.inf

    assert matrix.precision("C") == 0
    assert matrix.recall("C") == 0
    assert matrix.sensitivity("C") == 0
    assert matrix.specificity("C") == 0
    assert matrix.f1("C") == 0
    assert matrix.matthews_corrcoef("C") == np.inf


def test_classification_matrix_empty_lists():
    # Empty lists
    with pytest.raises(ValueError):
        ClassificationMatrix([], [])
    with pytest.raises(ValueError):
        ClassificationMatrix([1, 2, 3], [])
    with pytest.raises(ValueError):
        ClassificationMatrix([], [1, 2, 3])


def test_classification_matrix_two_classes_specified_symbol():
    # 12 cases for two classes
    # eight ones and 4 zeros
    targets = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    #
    predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    #
    matrix = ClassificationMatrix(predictions, targets, class_symbol=1)
    tp_symbol = 1
    tn_symbol = 0
    i = matrix.classes_lookup[tp_symbol]
    j = matrix.classes_lookup[tn_symbol]
    assert matrix.conf_matrix[i, i] == 6
    assert matrix.conf_matrix[j, j] == 3

    assert matrix.accuracy() == 9 / 12
    assert matrix.precision() == 6 / 7
    assert matrix.recall() == 6 / 8
    assert matrix.sensitivity() == 6 / 8
    assert matrix.specificity() == 3 / 4

    matrix2 = ClassificationMatrix(predictions, targets, class_symbol=0)
    tp_symbol = 0
    tn_symbol = 1
    i = matrix2.classes_lookup[tp_symbol]
    j = matrix2.classes_lookup[tn_symbol]
    assert matrix2.conf_matrix[i, i] == 3
    assert matrix2.conf_matrix[j, j] == 6

    assert matrix2.accuracy() == 9 / 12
    assert matrix2.precision() == 3 / 5
    assert matrix2.recall() == 3 / 4
    assert matrix2.sensitivity() == 3 / 4
    assert matrix2.specificity() == 6 / 8

    # 12 characters cases for two classes
    # eight 'A' and 4 'B'
    targets = ["A", "A", "A", "A", "A", "A", "A", "A", "B", "B", "B", "B"]
    #
    predictions = ["A", "A", "A", "A", "A", "B", "B", "A", "B", "B", "A", "B"]
    #

    matrix3 = ClassificationMatrix(predictions, targets, class_symbol="A")
    tp_symbol = "A"
    tn_symbol = "B"
    i = matrix3.classes_lookup[tp_symbol]
    j = matrix3.classes_lookup[tn_symbol]
    assert matrix3.conf_matrix[i, i] == 6
    assert matrix3.conf_matrix[j, j] == 3

    assert matrix3.accuracy() == 9 / 12
    assert matrix3.precision() == 6 / 7
    assert matrix3.recall() == 6 / 8
    assert matrix3.sensitivity() == 6 / 8
    assert matrix3.specificity() == 3 / 4


def test_classification_matrix_two_classes_different_len():
    # Increase the length of the lists
    for list_size in range(1, 5):
        targets = [0] * list_size
        predictions = [0] * list_size
        matrix = ClassificationMatrix(predictions, targets, class_symbol=1)
        assert matrix.conf_matrix[0, 0] == list_size

    # Increase the number of correct predictions
    targets = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    #
    predictions = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    for i in range(12):

        if i < 6:
            predictions[i] = 0
            tp = i + 1
            tn = 0
            fp = 6
            fn = 5 - i
        else:
            predictions[i] = 1
            tp = 6
            tn = i - 5
            fp = 11 - i
            fn = 0

        matrix = ClassificationMatrix(predictions, targets, class_symbol=0)
        assert matrix.accuracy() == (tp + tn) / (tp + tn + fp + fn)
        assert matrix.precision() == tp / (tp + fp)
        assert matrix.recall() == tp / (tp + fn)
        assert matrix.sensitivity() == tp / (tp + fn)
        assert matrix.specificity() == tn / (tn + fp)


def test_classification_matrix_three_classes():
    # 12 cases for three classes
    # 6 'A', 4 'B' and 2 'C'
    targets = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
    #
    predictions = ["A", "B", "B", "C", "A", "A", "B", "B", "C", "B", "C", "C"]
    #
    matrix = ClassificationMatrix(predictions, targets)
    tp_symbol = "A"
    i = matrix.classes_lookup[tp_symbol]
    assert matrix.conf_matrix[i, i] == 3

    assert matrix.accuracy() == 8 / 12
    assert matrix.precision() == 3 / (3 + 0)
    assert matrix.recall() == 3 / (3 + 3)
    assert matrix.sensitivity() == 3 / (3 + 3)
    assert matrix.specificity() == 5 / (5 + 0)

    assert matrix.precision(class_symbol="A") == 3 / (3 + 0)
    assert matrix.recall("A") == 3 / (3 + 3)
    assert matrix.sensitivity("A") == 3 / (3 + 3)
    assert matrix.specificity("A") == 5 / (5 + 0)

    assert matrix.precision(class_symbol="B") == 3 / (3 + 2)
    assert matrix.recall("B") == 3 / (3 + 1)
    assert matrix.sensitivity("B") == 3 / (3 + 1)
    assert matrix.specificity("B") == 5 / (5 + 2)

    assert matrix.precision(class_symbol="C") == 2 / (2 + 2)
    assert matrix.recall("C") == 2 / (2 + 0)
    assert matrix.sensitivity("C") == 2 / (2 + 0)
    assert matrix.specificity("C") == 6 / (6 + 2)


def test_classification_matrix_f1():
    targets = np.random.randint(2, size=20)
    # Shuffle the middle part of the targets
    section = targets[5:15]
    np.random.shuffle(section)
    predictions = np.r_[targets[:5], section, targets[15:]]
    #
    matrix = ClassificationMatrix(targets, predictions)
    precision = matrix.precision()
    recall = matrix.recall()
    assert matrix.f1() == (2 * precision * recall) / (precision + recall)


def test_classification_matrix_matthews_corrcoef():
    targets = np.random.randint(2, size=20)
    # Shuffle the middle part of the targets
    section = targets[5:15]
    np.random.shuffle(section)
    predictions = np.r_[targets[:5], section, targets[15:]]
    #
    matrix = ClassificationMatrix(targets, predictions)
    tp = matrix.tp()
    fp = matrix.fp()
    tn = matrix.tn()
    fn = matrix.fn()
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        assert matrix.matthews_corrcoef() == np.inf
    else:
        assert matrix.matthews_corrcoef() == (
            (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )

    # Check the zero denominator
    matrix = ClassificationMatrix([1, 1, 1, 0], [1, 1, 1, 1])
    assert matrix.matthews_corrcoef() == np.inf
