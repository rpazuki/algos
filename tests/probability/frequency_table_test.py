import pytest
import numpy as np
from numpy.testing import assert_array_equal
from probability.distributions import FrequencyTable


def test_empty_frequency_table():
    freq_table = FrequencyTable([])
    assert_array_equal(freq_table.np_keys(), [])
    assert freq_table["A"] == 0
    assert freq_table[3] == 0

    with pytest.raises(ValueError):
        FrequencyTable(None)


def test_single_element_frequency_table():
    # Four elements
    samples = ["A", "A", "A", "A"]
    freq_table = FrequencyTable(samples)
    assert_array_equal(freq_table.np_keys(), ["A"])
    assert freq_table.total == 4
    assert freq_table["A"] == 4
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [1])
    assert_array_equal(freq_table.frequencies(normalised=False), [4])


def test_two_elements_frequency_table():
    # Eight elements
    samples = ["A", "A", "A", "A", "A", "A", "B", "B"]
    freq_table = FrequencyTable(samples)
    assert_array_equal(freq_table.np_keys(), ["A", "B"])
    assert freq_table.total == 8
    assert freq_table["A"] == 6
    assert freq_table["B"] == 2
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [6 / 8, 2 / 8])
    assert_array_equal(freq_table.frequencies(normalised=False), [6, 2])


def test_three_elements_frequency_table():
    # 12 elements
    samples = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
    freq_table = FrequencyTable(samples)
    assert_array_equal(freq_table.np_keys(), ["A", "B", "C"])
    assert freq_table.total == 12
    assert freq_table["A"] == 6
    assert freq_table["B"] == 4
    assert freq_table["C"] == 2
    assert freq_table["D"] == 0
    assert_array_equal(
        freq_table.frequencies(normalised=True), [6 / 12, 4 / 12, 2 / 12]
    )
    assert_array_equal(freq_table.frequencies(normalised=False), [6, 4, 2])


def test_most_common_frequency_table():
    samples = np.r_[["A"] * 24, ["B"] * 48, ["C"] * 4, ["D"] * 7, ["E"] * 17]
    np.random.shuffle(samples)

    freq_table = FrequencyTable(samples)
    assert_array_equal(freq_table.most_common(1), [("B", 48)])
    assert_array_equal(freq_table.most_common(2), [("B", 48), ("A", 24)])
    assert_array_equal(freq_table.most_common(3), [("B", 48), ("A", 24), ("E", 17)])
    assert_array_equal(
        freq_table.most_common(4), [("B", 48), ("A", 24), ("E", 17), ("D", 7)]
    )
    assert_array_equal(
        freq_table.most_common(5), [("B", 48), ("A", 24), ("E", 17), ("D", 7), ("C", 4)]
    )
    assert_array_equal(
        freq_table.most_common(), [("B", 48), ("A", 24), ("E", 17), ("D", 7), ("C", 4)]
    )

    # Empty list
    freq_table = FrequencyTable([])
    assert_array_equal(freq_table.most_common(), [])
    assert_array_equal(freq_table.most_common(1), [])
    assert_array_equal(freq_table.most_common(2), [])


def test_from_dict_frequency_table():
    # None dict
    with pytest.raises(ValueError):
        FrequencyTable(None)

    # Empty dict
    freq_table = FrequencyTable({})
    assert_array_equal(freq_table.np_keys(), [])
    assert freq_table["A"] == 0
    assert freq_table[3] == 0

    # Single class
    freq_table = FrequencyTable({"A": 3})
    assert_array_equal(freq_table.np_keys(), ["A"])
    assert freq_table.total == 3
    assert freq_table["A"] == 3
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [1])
    assert_array_equal(freq_table.frequencies(normalised=False), [3])

    # Single class with zero sample
    freq_table = FrequencyTable({"A": 0})
    assert_array_equal(freq_table.np_keys(), ["A"])
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [0])
    assert_array_equal(freq_table.frequencies(normalised=False), [0])

    # Two.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4})
    assert_array_equal(freq_table.np_keys(), ["A", "B"])
    assert freq_table.total == 7
    assert freq_table["A"] == 3
    assert freq_table["B"] == 4
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [3 / 7, 4 / 7])
    assert_array_equal(freq_table.frequencies(normalised=False), [3, 4])

    # Two.classes() with zero sample
    freq_table = FrequencyTable({"A": 0, "B": 3})
    assert_array_equal(freq_table.np_keys(), ["A", "B"])
    assert freq_table.total == 3
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [0, 1])
    assert_array_equal(freq_table.frequencies(normalised=False), [0, 3])

    freq_table = FrequencyTable({"A": 0, "B": 0})
    assert_array_equal(freq_table.np_keys(), ["A", "B"])
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [0, 0])
    assert_array_equal(freq_table.frequencies(normalised=False), [0, 0])

    # Three.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4, "C": 4})
    assert_array_equal(freq_table.np_keys(), ["A", "B", "C"])
    assert freq_table.total == 11
    assert freq_table["A"] == 3
    assert freq_table["B"] == 4
    assert freq_table["C"] == 4
    assert freq_table[1] == 0
    assert_array_equal(
        freq_table.frequencies(normalised=True), [3 / 11, 4 / 11, 4 / 11]
    )
    assert_array_equal(freq_table.frequencies(normalised=False), [3, 4, 4])

    # Three.classes() with zero sample
    freq_table = FrequencyTable({"A": 0, "B": 3, "C": 3})
    assert_array_equal(freq_table.np_keys(), ["A", "B", "C"])
    assert freq_table.total == 6
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table["C"] == 3
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [0, 3 / 6, 3 / 6])
    assert_array_equal(freq_table.frequencies(normalised=False), [0, 3, 3])

    freq_table = FrequencyTable({"A": 0, "B": 0, "C": 0})
    assert_array_equal(freq_table.np_keys(), ["A", "B", "C"])
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table["C"] == 0
    assert freq_table[1] == 0
    assert_array_equal(freq_table.frequencies(normalised=True), [0, 0, 0])
    assert_array_equal(freq_table.frequencies(normalised=False), [0, 0, 0])


def test_add_frequency_table():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    freq_table3 = freq_table1 + freq_table2

    assert_array_equal(freq_table3.np_keys(), ["A", "B", "C", "D"])
    assert freq_table3.total == (freq_table1.total + freq_table2.total)
    assert freq_table3["A"] == 7
    assert freq_table3["B"] == 4
    assert freq_table3["C"] == 11
    assert freq_table3["D"] == 10
    assert_array_equal(
        freq_table3.frequencies(normalised=True), [7 / 32, 4 / 32, 11 / 32, 10 / 32]
    )
    assert_array_equal(freq_table3.frequencies(normalised=False), [7, 4, 11, 10])


def test_product_frequency_table():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    # different tables
    freq_table3 = freq_table1 * freq_table2
    print(freq_table3)
