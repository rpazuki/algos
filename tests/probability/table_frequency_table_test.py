from collections import Counter
import numpy as np
from probability import FrequencyTable


def test_constructor_frequency_table():

    table = FrequencyTable({"one": 1, "two": 2, "three": 3}, names=["Y1"])
    assert table["one"] == 1
    assert table["two"] == 2
    assert table["three"] == 3

    table = FrequencyTable([("two", 2), ("one", 1), ("three", 3)])
    assert table[("two", 2)] == 1
    assert table["one", 1] == 1
    assert table["three", 3] == 1

    samples = [1, 2, 2, 3, 3, 3]
    counter = Counter(samples)
    table = FrequencyTable(counter)
    assert table[1] == 1
    assert table[2] == 2
    assert table[3] == 3

    table = FrequencyTable(zip(["one", "two", "three"], [1, 2, 3]))
    assert table["one", 1] == 1
    assert table["two", 2] == 1
    assert table["three", 3] == 1

    # numeric key
    table = FrequencyTable({1: 1, 2: 2, 3: 3})
    assert table[1] == 1
    assert table[2] == 2
    assert table[3] == 3

    table = FrequencyTable({1.1: 1, 2.2: 2, 3.3: 3})
    assert table[1.1] == 1
    assert table[2.2] == 2
    assert table[3.3] == 3


def test_from_np_array_frequency_table():
    arr = np.r_[np.zeros(40), np.ones(60)]
    table = FrequencyTable.from_np_array(arr, ["X1"])
    assert table[0] == 40
    assert table[1] == 60

    assert table.frequency(0) == 40
    assert table.frequency(1) == 60

    assert table.freq(X1=0) == 40
    assert table.freq(X1=1) == 60

    assert table.probability(0) == 0.40
    assert table.probability(1) == 0.60

    assert table.prob(X1=0) == 0.40
    assert table.prob(X1=1) == 0.60

    arr = np.r_[np.zeros((40, 2)), np.ones((60, 2))]
    table = FrequencyTable.from_np_array(arr, ["X1", "X2"])
    assert table[0, 0] == 40
    assert table[1, 1] == 60

    assert table.frequency((0, 0)) == 40
    assert table.frequency((1, 1)) == 60

    assert table.freq(X1=0, X2=0) == 40
    assert table.freq(X1=1, X2=1) == 60

    assert table.probability((0, 0)) == 0.40
    assert table.probability((1, 1)) == 0.60

    assert table.prob(X1=0, X2=0) == 0.40
    assert table.prob(X1=1, X2=1) == 0.60
