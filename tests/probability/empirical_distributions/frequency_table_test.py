import pytest
from pytest import approx
import numpy as np
from probability import Distribution
from probability.empirical_distributions import FrequencyTable
from probability.empirical_distributions import EmpiricalDistribution
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
    assert all(compare(freq_table.levels(), ["A"]))
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
    assert all(compare(freq_table.levels(), [(1,)]))
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
    assert all(compare(freq_table.levels(), [("A",)]))
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
    assert all(compare(freq_table.levels(), [(1, 2), (1, 3)]))
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
    assert all(compare(freq_table.levels(), [("A", "B"), ("A", "C")]))
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
    assert all(compare(freq_table.levels(), ["A", "B"]))
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
    assert all(compare(freq_table.levels(), ["A", "B", "C"]))
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
    assert all(compare(freq_table.levels(), ["A", "B", "C", "D", "E"]))
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
    assert all(compare(freq_table.levels(), []))
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
    assert all(compare(freq_table.levels(), []))
    assert freq_table["A"] == 0
    assert freq_table[3] == 0

    # Single class
    freq_table = FrequencyTable({"A": 3})
    assert all(compare(freq_table.keys_as_list(), ["A"]))
    assert all(compare(freq_table.levels(), ["A"]))
    assert freq_table.total == 3
    assert freq_table["A"] == 3
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [1]))
    assert all(compare(freq_table.frequencies(normalised=False), [3]))

    # Single class with zero sample
    freq_table = FrequencyTable({"A": 0})
    assert all(compare(freq_table.keys_as_list(), ["A"]))
    assert all(compare(freq_table.levels(), ["A"]))
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0]))
    assert all(compare(freq_table.frequencies(normalised=False), [0]))

    # Two.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert all(compare(freq_table.levels(), ["A", "B"]))
    assert freq_table.total == 7
    assert freq_table["A"] == 3
    assert freq_table["B"] == 4
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [3 / 7, 4 / 7]))
    assert all(compare(freq_table.frequencies(normalised=False), [3, 4]))

    # Two.classes() with zero sample
    freq_table = FrequencyTable({"A": 0, "B": 3})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert all(compare(freq_table.levels(), ["A", "B"]))
    assert freq_table.total == 3
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 1]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 3]))

    freq_table = FrequencyTable({"A": 0, "B": 0})
    assert all(compare(freq_table.keys_as_list(), ["A", "B"]))
    assert all(compare(freq_table.levels(), ["A", "B"]))
    assert freq_table.total == 0
    assert freq_table["A"] == 0
    assert freq_table["B"] == 0
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 0]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 0]))

    # Three.classes()
    freq_table = FrequencyTable({"A": 3, "B": 4, "C": 4})
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert all(compare(freq_table.levels(), ["A", "B", "C"]))
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
    assert all(compare(freq_table.levels(), ["A", "B", "C"]))
    assert freq_table.total == 6
    assert freq_table["A"] == 0
    assert freq_table["B"] == 3
    assert freq_table["C"] == 3
    assert freq_table[1] == 0
    assert all(compare(freq_table.frequencies(normalised=True), [0, 3 / 6, 3 / 6]))
    assert all(compare(freq_table.frequencies(normalised=False), [0, 3, 3]))

    freq_table = FrequencyTable({"A": 0, "B": 0, "C": 0})
    assert all(compare(freq_table.keys_as_list(), ["A", "B", "C"]))
    assert all(compare(freq_table.levels(), ["A", "B", "C"]))
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
    assert all(compare(freq_table3.levels(), ["A", "B", "C", "D"]))
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
    both_levels = zip(freq_table3.levels(), [["A", "B", "C"], ["A", "C", "D"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))

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


def test_product_multi_proc_frequency_table():
    sample_1 = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "U": 21,
        "V": 22,
        "W": 23,
        "X": 24,
        "Y": 25,
        "Z": 26,
    }
    freq_table1 = FrequencyTable(sample_1, "X1")
    freq_table2 = FrequencyTable(sample_1, "X2")

    prod_1 = freq_table1 * freq_table2

    prod_2 = freq_table1.product_multi_proc(freq_table2, 2)
    for key, value in prod_1.items():
        assert prod_2[key] == value

    prod_2 = freq_table1.product_multi_proc(freq_table2, 3)
    for key, value in prod_1.items():
        assert prod_2[key] == value

    prod_2 = freq_table1.product_multi_proc(freq_table2, 4)
    for key, value in prod_1.items():
        assert prod_2[key] == value

    prod_2 = freq_table1.product_multi_proc(freq_table2, 5)
    for key, value in prod_1.items():
        assert prod_2[key] == value

    prod_2 = freq_table1.product_multi_proc(freq_table2, 6)
    for key, value in prod_1.items():
        assert prod_2[key] == value


def test_statistical_independence_frequency_table():
    # P(x,y,z) = P(x)P(y)P(z)
    # to check that, first, create a joint dist. by product
    # then marginalis and multiply again. The final must be equal
    # the joint
    freq_table1 = FrequencyTable({"A": 1, "B": 2, "C": 3}, name="X")
    freq_table2 = FrequencyTable({"a": 4, "b": 5, "c": 6}, name="Y")
    freq_table3 = FrequencyTable({11: 7, 22: 8, 33: 9}, name="Z")

    joint_dist = freq_table1 * freq_table2 * freq_table3

    px_marginal = joint_dist.marginal("Y", "Z")
    py_marginal = joint_dist.marginal("X", "Z")
    pz_marginal = joint_dist.marginal("X", "Y")

    joint_dist2 = px_marginal * py_marginal * pz_marginal

    for k1 in joint_dist:
        assert joint_dist.probability(k1) == joint_dist2.probability(k1)


def test_continouse_frequency_table():
    # Note that there are three cases at the top
    # of the 'samples' that are out of the range
    # of the bins
    samples = [
        -0.8,
        -0.1,
        1.1,
        0.08706673,
        0.48376282,
        0.52239421,
        0.14593262,
        0.07347176,
        0.34733567,
        0.280569,
        0.28010016,
        0.00394102,
        0.68676722,
        0.91315035,
        0.79438912,
        0.73380882,
        0.75251795,
        0.87636918,
        0.42696308,
        0.42906385,
        0.2679933,
        0.49831989,
        0.76442673,
        0.70112504,
        0.01672044,
        0.88090148,
        0.69801565,
        0.27066378,
        0.93762043,
        0.45260394,
        0.13722068,
        0.35406184,
        0.27922478,
    ]
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    digitized = EmpiricalDistribution.digitize_bin(samples, bins)
    dist = FrequencyTable(digitized)
    assert dist.total == 33
    assert dist[0.0] == 4
    assert dist.prob(X1=0.0) == 4 / 33
    assert dist[0.8] == 2
    assert dist.prob(X1=0.8) == 2 / 33
    assert dist[0.9] == 2
    assert dist.prob(X1=0.9) == 2 / 33
    # These three levels are out of the bins range
    # belong to two elements on top
    assert dist[-0.1] == 2
    assert dist.prob(X1=-0.1) == 2 / 33
    assert dist[1.0] == 1
    assert dist.prob(X1=1.0) == 1 / 33

    digitized = EmpiricalDistribution.digitize(samples, start=0, stop=1, num=11)
    dist = FrequencyTable(digitized)
    assert dist.total == 33
    assert dist[0.0] == 4
    assert dist.prob(X1=0.0) == 4 / 33
    assert dist[0.8] == 2
    assert dist.prob(X1=0.8) == 2 / 33
    assert dist[0.9] == 2
    assert dist.prob(X1=0.9) == 2 / 33
    # These three levels are out of the bins range
    # belong to two elements on top
    assert dist[-0.1] == 2
    assert dist.prob(X1=-0.1) == 2 / 33
    assert dist[1.0] == 1
    assert dist.prob(X1=1.0) == 1 / 33


def test_continouse_level_frequency_table():
    samples = [15.23, 9.7, 13.78, -1, 999, 12.8, 2.5, 6.35, 14.3, 16.3]
    bins = [0, 10, 20]
    levels = ["low", "high"]
    digitized = EmpiricalDistribution.digitize_bin(samples, bins, levels=levels)
    dist = FrequencyTable(digitized)
    assert dist["less"] == 1
    assert dist.prob(X1="less") == 0.1
    assert dist["low"] == 3
    assert dist.prob(X1="low") == 0.3
    assert dist["high"] == 5
    assert dist.prob(X1="high") == 0.5
    assert dist["more"] == 1
    assert dist.prob(X1="more") == 0.1

    digitized = EmpiricalDistribution.digitize(
        samples, start=0, stop=20, num=3, levels=levels
    )
    dist = FrequencyTable(digitized)
    assert dist["less"] == 1
    assert dist.prob(X1="less") == 0.1
    assert dist["low"] == 3
    assert dist.prob(X1="low") == 0.3
    assert dist["high"] == 5
    assert dist.prob(X1="high") == 0.5
    assert dist["more"] == 1
    assert dist.prob(X1="more") == 0.1


def test_levels_is_numeric_frequency_table():
    samples = [1, 2, 3, 4, 5, 6]
    dist = FrequencyTable(samples)
    assert dist.discrete_rv.is_numeric

    samples = [1.1, 2.6, 3.6, 4.9, 5.6, 6.7]
    dist = FrequencyTable(samples)
    assert dist.discrete_rv.is_numeric

    samples = ["A", "B", "C", "D"]
    dist = FrequencyTable(samples)
    assert not dist.discrete_rv.is_numeric


def test_avg_frequency_table():
    samples = ["A", "B", "C", "D"]
    dist = FrequencyTable(samples)
    with pytest.raises(TypeError):
        dist.avg()

    assert dist.avg(operation=lambda values: [10] * len(values)) == 10
    assert dist.std(operation=lambda values: [10] * len(values)) == 0
    assert dist.moment(order=3, operation=lambda values: [10] * len(values)) == 1000
    assert dist.avg(operation=lambda values: [1, 2, 3, 4]) == 2.5
    assert dist.std(operation=lambda values: [1, 2, 3, 4]) == 1.25
    assert dist.moment(order=3, operation=lambda values: [1, 2, 3, 4]) == 100 / 4

    samples = [1, 2, 3, 4]
    dist = FrequencyTable(samples)
    assert dist.avg() == 2.5
    assert dist.std() == 1.25
    assert dist.moment(3) == 25
    samples = [1, 2, 3, 4, 4]
    dist = FrequencyTable(samples)
    assert dist.avg() == 14 / 5
    assert dist.std() == approx(1.36)
    assert dist.moment(order=3) == 164 / 5
    samples = [1, 2, 3, 4, 4, 5]
    dist = FrequencyTable(samples)
    assert dist.avg() == 19 / 6
    assert dist.std() == approx(1.8055555555556)
    assert dist.moment(order=3) == 289 / 6

    samples = [
        "high",
        "low",
        "low",
        "low",
        "high",
        "high",
        "high",
        "high",
        "high",
        "high",
        "low",
    ]

    def to_number(levels):
        return (1 if level == "high" else 2 for level in levels)

    dist = FrequencyTable(samples)
    assert dist.avg(to_number) == 15 / 11
    assert dist.std(to_number) == approx(0.23140495867769)


def test_keys_consistencies_frequency_table():
    with pytest.raises(ValueError):
        FrequencyTable([1, 2, 3, "A"], ["X1"], consistencies=True)

    with pytest.raises(ValueError):
        FrequencyTable(["A", 1, 2, 3], ["X1"], consistencies=True)

    with pytest.raises(ValueError):
        FrequencyTable([(1,), (2,), (3,), (4, 5)], ["X1"], consistencies=True)
    with pytest.raises(ValueError):
        FrequencyTable([(4, 5), (1,), (2,), (3,)], ["X1"], consistencies=True)
    with pytest.raises(ValueError):
        FrequencyTable([(4, 5), (1, 3), (2, 3, 4), (3, 7)], ["X1"], consistencies=True)
