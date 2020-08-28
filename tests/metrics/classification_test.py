import pytest
from metrics.classification import ClassificationMatrix
import numpy as np
from numpy.testing import assert_array_equal


def test_classification_matrix_two_classes_first_symbol():
    # 12 cases for two classes
    # eight ones and 4 zeros
    targets = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    #
    predictions = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    #
    m = ClassificationMatrix(predictions, targets)
    TP_Symbol = 1
    TN_Symbol = 0
    i = m.classes_lookup[TP_Symbol]
    j = m.classes_lookup[TN_Symbol]
    assert m.conf_matrix[i, i] == 6
    assert m.conf_matrix[j, j] == 3

    assert m.accuracy() == 9/12
    assert m.precision() == 6/7
    assert m.recall() == 6/8
    assert m.sensitivity() == 6/8
    assert m.specificity() == 3/4


def test_classification_matrix_mismatch_in_class_names():
    targets = ['A'] * 2 + ['B'] * 3
    predictions = ['C'] * 2 + ['D'] * 3
    m = ClassificationMatrix(targets, predictions)
    assert m.TP('A') == 0
    assert m.TN('A') == 0
    assert m.FP('A') == 2
    assert m.FN('A') == 0

    assert m.TP('B') == 0
    assert m.TN('B') == 0
    assert m.FP('B') == 3
    assert m.FN('B') == 0

    assert m.TP('C') == 0
    assert m.TN('C') == 0
    assert m.FP('C') == 0
    assert m.FN('C') == 2

    assert m.TP('D') == 0
    assert m.TN('D') == 0
    assert m.FP('D') == 0
    assert m.FN('D') == 3

    assert m.accuracy() == 0

    assert m.precision('A') == 0
    assert m.recall('A') == 0
    assert m.sensitivity('A') == 0
    assert m.specificity('A') == 0
    assert m.f1('A') == 0
    assert m.matthews_corrcoef('A') == np.inf

    assert m.precision('C') == 0
    assert m.recall('C') == 0
    assert m.sensitivity('C') == 0
    assert m.specificity('C') == 0
    assert m.f1('C') == 0
    assert m.matthews_corrcoef('C') == np.inf


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
    m = ClassificationMatrix(predictions, targets, classSymbol=1)
    TP_Symbol = 1
    TN_Symbol = 0
    i = m.classes_lookup[TP_Symbol]
    j = m.classes_lookup[TN_Symbol]
    assert m.conf_matrix[i, i] == 6
    assert m.conf_matrix[j, j] == 3

    assert m.accuracy() == 9/12
    assert m.precision() == 6/7
    assert m.recall() == 6/8
    assert m.sensitivity() == 6/8
    assert m.specificity() == 3/4

    m2 = ClassificationMatrix(predictions, targets, classSymbol=0)
    TP_Symbol = 0
    TN_Symbol = 1
    i = m2.classes_lookup[TP_Symbol]
    j = m2.classes_lookup[TN_Symbol]
    assert m2.conf_matrix[i, i] == 3
    assert m2.conf_matrix[j, j] == 6

    assert m2.accuracy() == 9/12
    assert m2.precision() == 3/5
    assert m2.recall() == 3/4
    assert m2.sensitivity() == 3/4
    assert m2.specificity() == 6/8

    # 12 characters cases for two classes
    # eight 'A' and 4 'B'
    targets = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    #
    predictions = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'A', 'B']
    #

    m3 = ClassificationMatrix(predictions, targets, classSymbol='A')
    TP_Symbol = 'A'
    TN_Symbol = 'B'
    i = m3.classes_lookup[TP_Symbol]
    j = m3.classes_lookup[TN_Symbol]
    assert m3.conf_matrix[i, i] == 6
    assert m3.conf_matrix[j, j] == 3

    assert m3.accuracy() == 9/12
    assert m3.precision() == 6/7
    assert m3.recall() == 6/8
    assert m3.sensitivity() == 6/8
    assert m3.specificity() == 3/4


def test_classification_matrix_two_classes_different_len():
    # Increase the length of the lists
    for list_size in range(1, 5):
        targets = [0]*list_size
        predictions = [0]*list_size
        m = ClassificationMatrix(predictions, targets, classSymbol=1)
        assert m.conf_matrix[0, 0] == list_size

    # Increase the number of correct predictions
    targets = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    #
    predictions = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    for i in range(12):

        if i < 6:
            predictions[i] = 0
            tp = i+1
            tn = 0
            fp = 6
            fn = 5 - i
        else:
            predictions[i] = 1
            tp = 6
            tn = i-5
            fp = 11 - i
            fn = 0

        m = ClassificationMatrix(predictions, targets, classSymbol=0)
        assert m.accuracy() == (tp+tn)/(tp+tn+fp+fn)
        assert m.precision() == tp/(tp+fp)
        assert m.recall() == tp/(tp+fn)
        assert m.sensitivity() == tp/(tp+fn)
        assert m.specificity() == tn/(tn+fp)


def test_classification_matrix_Three_classes():
    # 12 cases for three classes
    # 6 'A', 4 'B' and 2 'C'
    targets = ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C']
    #
    predictions = ['A', 'B', 'B', 'C', 'A', 'A', 'B', 'B', 'C', 'B', 'C', 'C']
    #
    m = ClassificationMatrix(predictions, targets)
    TP_Symbol = 'A'
    i = m.classes_lookup[TP_Symbol]
    assert m.conf_matrix[i, i] == 3

    assert m.accuracy() == 8/12
    assert m.precision() == 3/(3+0)
    assert m.recall() == 3/(3+3)
    assert m.sensitivity() == 3/(3+3)
    assert m.specificity() == 5/(5+0)

    assert m.precision(classSymbol='A') == 3/(3+0)
    assert m.recall('A') == 3/(3+3)
    assert m.sensitivity('A') == 3/(3+3)
    assert m.specificity('A') == 5/(5+0)

    assert m.precision(classSymbol='B') == 3/(3+2)
    assert m.recall('B') == 3/(3+1)
    assert m.sensitivity('B') == 3/(3+1)
    assert m.specificity('B') == 5/(5+2)

    assert m.precision(classSymbol='C') == 2/(2+2)
    assert m.recall('C') == 2/(2+0)
    assert m.sensitivity('C') == 2/(2+0)
    assert m.specificity('C') == 6/(6+2)


def test_classification_matrix_f1():
    targets = np.random.randint(2, size=20)
    # Shuffle the middle part of the targets
    section = targets[5:15]
    np.random.shuffle(section)
    predictions = np.r_[targets[:5], section, targets[15:]]
    #
    m = ClassificationMatrix(targets, predictions)
    p = m.precision()
    r = m.recall()
    assert m.f1() == (2*p*r)/(p+r)


def test_classification_matrix_matthews_corrcoef():
    targets = np.random.randint(2, size=20)
    # Shuffle the middle part of the targets
    section = targets[5:15]
    np.random.shuffle(section)
    predictions = np.r_[targets[:5], section, targets[15:]]
    #
    m = ClassificationMatrix(targets, predictions)
    tp = m.TP()
    fp = m.FP()
    tn = m.TN()
    fn = m.FN()
    if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) == 0:
        assert m.matthews_corrcoef() == np.inf
    else:
        assert m.matthews_corrcoef() == (
            (tp*tn - fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            )

    # Check the zero denominator
    m = ClassificationMatrix([1, 1, 1, 0], [1, 1, 1, 1])
    assert m.matthews_corrcoef() == np.inf
