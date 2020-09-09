import pytest
import numpy as np
from probability.distributions import FrequencyTable
from tests.helpers import compare


def test_empty_frequency_table():
    freq_table = FrequencyTable([])
    assert all(compare(freq_table.keys_as_list(), []))
    assert freq_table["A"] == 0
    assert freq_table[3] == 0

    with pytest.raises(ValueError):
        FrequencyTable(None)


def test_single_element_frequency_table():
    # Four elements
    samples = ["A", "A", "A", "A"]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), ["A"]))
    assert freq_table.total == 4
    assert freq_table["A"] == 4
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [1]))
    assert all(compare(freq_table.frequencies(normalised=False), [4]))
    assert freq_table.prob("A") == 1
    assert freq_table.prob(X1="A") == 1
    assert freq_table.prob("B") == 0


def test_single_element_tuple_frequency_table():
    samples = [(1,), (1,), (1,), (1,)]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), [(1,)]))
    assert freq_table.total == 4
    assert freq_table[(1,)] == 4
    assert freq_table[(2,)] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [1]))
    assert all(compare(freq_table.frequencies(normalised=False), [4]))
    assert freq_table.prob((1,)) == 1
    assert freq_table.prob(X1=(1,)) == 1
    assert freq_table.prob((2,)) == 0

    samples = [("A",), ("A",), ("A",), ("A",)]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), [("A",)]))
    assert freq_table.total == 4
    assert freq_table[("A",)] == 4
    assert freq_table[("B",)] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [1]))
    assert all(compare(freq_table.frequencies(normalised=False), [4]))
    assert freq_table.prob(("A",)) == 1
    assert freq_table.prob(X1=("A",)) == 1
    assert freq_table.prob(("B",)) == 0


def test_double_elements_tuple_frequency_table():
    samples = [(1, 2), (1, 2), (1, 2), (1, 3)]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), [(1, 2), (1, 3)]))
    assert freq_table.total == 4
    assert freq_table[(1, 2)] == 3
    assert freq_table[(1, 3)] == 1
    assert freq_table[1] == 0
    assert freq_table[2] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0.75, 0.25]))
    assert all(compare(freq_table.frequencies(normalised=False), [3, 1]))
    assert freq_table.prob((1, 2)) == 3 / 4
    assert freq_table.prob((1, 3)) == 1 / 4
    assert freq_table.prob(X1=(1, 2)) == 3 / 4
    assert freq_table.prob(X1=(1, 3)) == 1 / 4

    samples = [("A", "B"), ("A", "B"), ("A", "C"), ("A", "C")]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), [("A", "B"), ("A", "C")]))
    assert freq_table.total == 4
    assert freq_table[("A", "B")] == 2
    assert freq_table[("A", "C")] == 2
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0.5, 0.5]))
    assert all(compare(freq_table.frequencies(normalised=False), [2, 2]))
    assert freq_table.prob(("A", "B")) == 2 / 4
    assert freq_table.prob(("A", "C")) == 2 / 4
    assert freq_table.prob(X1=("A", "B")) == 2 / 4
    assert freq_table.prob(X1=("A", "C")) == 2 / 4


def test_two_elements_frequency_table():
    # Eight elements
    samples = ["A", "A", "A", "A", "A", "A", "B", "B"]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert freq_table.total == 8
    assert freq_table["A"] == 6
    assert freq_table["B"] == 2
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [6 / 8, 2 / 8]))
    assert all(compare(freq_table.frequencies(normalised=False), [6, 2]))
    assert freq_table.prob("A") == 6 / 8
    assert freq_table.prob("B") == 2 / 8
    assert freq_table.prob("C") == 0
    assert freq_table.prob(X1="A") == 6 / 8
    assert freq_table.prob(X1="B") == 2 / 8


def test_three_elements_frequency_table():
    # 12 elements
    samples = ["A", "A", "A", "A", "A", "A", "B", "B", "B", "B", "C", "C"]
    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert freq_table.total == 12
    assert freq_table["A"] == 6
    assert freq_table["B"] == 4
    assert freq_table["C"] == 2
    assert freq_table["D"] == 0
    assert all(
        compare(freq_table.frequencies(normalised=True), [6 / 12, 4 / 12, 2 / 12])
    )
    assert all(compare(freq_table.frequencies(normalised=False), [6, 4, 2]))
    assert freq_table.prob("A") == 6 / 12
    assert freq_table.prob("B") == 4 / 12
    assert freq_table.prob("C") == 2 / 12
    assert freq_table.prob("D") == 0
    assert freq_table.prob(X1="A") == 6 / 12
    assert freq_table.prob(X1="B") == 4 / 12
    assert freq_table.prob(X1="C") == 2 / 12
    assert freq_table.prob(X1="D") == 0


def test_most_common_frequency_table():
    samples = np.r_[["A"] * 24, ["B"] * 48, ["C"] * 4, ["D"] * 7, ["E"] * 17]
    np.random.shuffle(samples)

    freq_table = FrequencyTable(samples)
    assert all(compare(freq_table.most_common(1), [("B", 48)]))
    assert all(compare(freq_table.most_common(2), [("B", 48), ("A", 24)]))
    assert all(compare(freq_table.most_common(3), [("B", 48), ("A", 24), ("E", 17)]))
    assert all(
        compare(freq_table.most_common(4), [("B", 48), ("A", 24), ("E", 17), ("D", 7)])
    )
    assert all(
        compare(
            freq_table.most_common(5),
            [("B", 48), ("A", 24), ("E", 17), ("D", 7), ("C", 4)],
        )
    )
    assert all(
        compare(
            freq_table.most_common(),
            [("B", 48), ("A", 24), ("E", 17), ("D", 7), ("C", 4)],
        )
    )

    # Empty list
    freq_table = FrequencyTable([])
    assert all(compare(freq_table.most_common(), []))
    assert all(compare(freq_table.most_common(1), []))
    assert all(compare(freq_table.most_common(2), []))


def test_from_dict_frequency_table():
    # None dict
    with pytest.raises(ValueError):
        FrequencyTable(None)

    # Empty dict
    freq_table = FrequencyTable({})
    assert all(compare(freq_table.keys_as_list(), []))
    assert freq_table["A"] == 0
    assert freq_table[3] == 0

    # Single class
    freq_table = FrequencyTable({"A": 3})
    assert all(compare(freq_table.keys_as_list(), ["A"]))
    assert freq_table.total == 3
    assert freq_table["A"] == 3
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [1]))
    assert all(compare(freq_table.frequencies(normalised=False), [3]))

    # Single class with zero sample
    freq_table = FrequencyTable({"A": 0})
    assert all(compare(freq_table.keys_as_list(), ["A"]))
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0]))
    assert all(compare(freq_table.frequencies(normalised=False), [0]))

    # Two.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert freq_table.total == 7
    assert freq_table["A"] == 3
    assert freq_table["B"] == 4
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [3 / 7, 4 / 7]))
    assert all(compare(freq_table.frequencies(normalised=False), [3, 4]))

    # Two.classes() with zero sample
    freq_table = FrequencyTable({"A": 0, "B": 3})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert freq_table.total == 3
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 1]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 3]))

    freq_table = FrequencyTable({"A": 0, "B": 0})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 0]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 0]))

    # Three.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4, "C": 4})
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert freq_table.total == 11
    assert freq_table["A"] == 3
    assert freq_table["B"] == 4
    assert freq_table["C"] == 4
    assert freq_table[1] == 0
    assert all(
        compare(freq_table.frequencies(normalised=True), [3 / 11, 4 / 11, 4 / 11])
    )
    assert all(compare(freq_table.frequencies(normalised=False), [3, 4, 4]))

    # Three.classes() with zero sample
    freq_table = FrequencyTable({"A": 0, "B": 3, "C": 3})
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert freq_table.total == 6
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table["C"] == 3
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 3 / 6, 3 / 6]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 3, 3]))

    freq_table = FrequencyTable({"A": 0, "B": 0, "C": 0})
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table["C"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 0, 0]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 0, 0]))


def test_from_iterator_frequency_table():
    samples = iter([1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    freq_table = FrequencyTable(samples)

    assert freq_table["A"] == 0
    assert freq_table[1] == 6
    assert freq_table[2] == 4


def test_add_frequency_table():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    freq_table3 = freq_table1 + freq_table2

    assert all(compare(freq_table3.keys_as_list(), ["A", "B", "C", "D"]))
    assert freq_table3.total == (freq_table1.total + freq_table2.total)
    assert freq_table3["A"] == 7
    assert freq_table3["B"] == 4
    assert freq_table3["C"] == 11
    assert freq_table3["D"] == 10
    assert all(
        compare(
            freq_table3.frequencies(normalised=True), [7 / 32, 4 / 32, 11 / 32, 10 / 32]
        )
    )
    assert all(compare(freq_table3.frequencies(normalised=False), [7, 4, 11, 10]))


def test_product_exceptions_frequency_table():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    with pytest.raises(ValueError):
        freq_table1.product(2)


def test_product_frequency_table():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    # different tables, same name
    freq_table3 = freq_table1 * freq_table2
    assert all(compare(freq_table3.names, ["X11", "X12"]))
    assert all(compare(freq_table3.rvs.levels, [{"A", "B", "C"}, {"A", "D", "C"}]))

    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    # different tables, different names
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7}, name="X")
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10}, name="Y")
    freq_table3 = freq_table1 * freq_table2
    assert all(compare(freq_table3.names, ["X", "Y"]))
    freq_table3 = freq_table2 * freq_table1
    assert all(compare(freq_table3.names, ["Y", "X"]))

    # the same table
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7}, name="X")
    freq_table3 = freq_table1 * freq_table1
    assert all(compare(freq_table3.names, ["X1", "X2"]))

    # check the probabilites
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7})
    freq_table2 = FrequencyTable({"A": 4, "C": 4, "D": 10})

    freq_table3 = freq_table1 * freq_table2
    assert freq_table3.total == (freq_table1.total * freq_table2.total)
    assert freq_table3.probability(("A", "C")) == freq_table1.probability(
        "A"
    ) * freq_table2.probability("C")


def test_statistical_independence_frequency_table():
    # P(x,y,z) = P(x)P(y)P(z)
    # to check that, first, create a joint dist. by product
    # then marginalis and multiply again. The final must be equal
    # the joint
    freq_table1 = FrequencyTable({"A": 1, "B": 2, "C": 3}, name="X")
    freq_table2 = FrequencyTable({"a": 4, "b": 5, "c": 6}, name="Y")
    freq_table3 = FrequencyTable({11: 7, 22: 8, 33: 9}, name="Z")

    joint_dist = freq_table1 * freq_table2 * freq_table3

    px_marginal = joint_dist.marginal(["Y", "Z"])
    py_marginal = joint_dist.marginal(["X", "Z"])
    pz_marginal = joint_dist.marginal(["X", "Y"])

    joint_dist2 = px_marginal * py_marginal * pz_marginal

    for k1 in joint_dist:
        assert joint_dist.probability(k1) == joint_dist2.probability(k1)
