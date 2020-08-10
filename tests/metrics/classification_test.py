from metrics import classification as cs
import numpy as np
from numpy.testing import assert_array_equal


def test_confusion_matrix_two_classes():
    # 12 cases for two classes
    # eight ones and 4 zeros
    targets = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    #
    outputs = [1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
    #
    conf_matrix = np.asarray(([[3, 1], [2, 6]]))

    m = cs.ConfusionMatrix(outputs, targets)
    assert_array_equal(m.conf_matrix, conf_matrix)

    assert m.accuracy() == 9/12
    assert m.precision() == 6/7
    assert m.recall() == 6/8
    assert m.sensitivity() == 6/8
    assert m.specificity() == 3/4
