import pytest
from pytest import approx
import numpy as np
from probability.distributions import DiscreteDistribution
from tests.helpers import compare


def test_keys_consistencies_discrete_distribution():
    with pytest.raises(ValueError):
        DiscreteDistribution([1, 2, 3, "A"], ["X1"], check_keys_consistencies=True)

    with pytest.raises(ValueError):
        DiscreteDistribution(["A", 1, 2, 3], ["X1"], check_keys_consistencies=True)

    with pytest.raises(ValueError):
        DiscreteDistribution(
            [(1,), (2,), (3,), (4, 5)], ["X1"], check_keys_consistencies=True
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [(4, 5), (1,), (2,), (3,)], ["X1"], check_keys_consistencies=True
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [(4, 5), (1, 3), (2, 3, 4), (3, 7)], ["X1"], check_keys_consistencies=True
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", "1", "w1"), ("b", 2, "w1"), ("c", 3, "w2"), ("d", 4, "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1"), ("b", "2", "w1"), ("c", 3, "w2"), ("d", 4, "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1"), ("b", 2, "w1"), ("c", 3, "w2"), ("d", "4", "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1"), ("b", 2), ("c", 3, "w2"), ("d", "4", "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [(1, "w1"), ("b", 2, "w1"), ("c", 3, "w2"), ("d", "4", "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", None, "w1"), ("b", 2, "w1"), ("c", 3, "w2"), ("d", "4", "w2")],
            ["X1", "X2", "X3"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", None), ("b", 2, "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", 4), ("b", None, "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", "4"), ("b", 2, "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, 1, 4), ("b", 2, "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", 4), ("b", "2", "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", 4), (1, 2, "w1", 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", 4), ("b", 2, "w1", 2), ("c", "3", "w2", "1")],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )
    with pytest.raises(ValueError):
        DiscreteDistribution(
            [("a", 1, "w1", 4), ("b", 2, 2), ("c", 3, "w2", 1)],
            ["X1", "X2", "X3", "X4"],
            check_keys_consistencies=True,
        )


def test_iterable_samples_discrete_distribution():
    samples = """It is a long established fact that a reader will be
     distracted by the readable content of a page when looking at its
     layout. The point of using Lorem Ipsum is that it has a more-or-less
     normal distribution of letters, as opposed to using 'Content here,
     content here', making it look like readable English."""
    dist = DiscreteDistribution(iter(samples))
    assert dist.total == len(samples)

    gen = (c for c in samples)
    dist = DiscreteDistribution(gen)
    assert dist.total == len(samples)


def test_numpy_array_discrete_distribution():
    # It is not numpy array
    with pytest.raises(ValueError):
        DiscreteDistribution.from_np_array({1, 2, 3})

    # It is not list of list
    with pytest.raises(ValueError):
        DiscreteDistribution.from_np_array([1, 2, 3])

    # list of list or numpy 2D array converts to tuples
    samples = np.r_[["A"] * 24, ["B"] * 48, ["C"] * 4, ["D"] * 7, ["E"] * 17]
    samples = samples.reshape((samples.shape[0], 1))
    dist = DiscreteDistribution.from_np_array(samples)
    assert dist.total == 100
    # It is important to use tuple as key
    # since list is not hashable
    assert dist.probability(("A",)) == 0.24
    assert dist.prob(X1=("A",)) == 0.24


def test_one_levels_discrete_distribution():
    dist = DiscreteDistribution({"Dog": 2})
    assert all(compare(dist.keys_as_list(), ["Dog"]))
    assert dist.rvs.size == 1
    assert dist["Dog"] == 2
    assert dist["Cat"] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob("Dog") == 1
    assert dist.prob(X1="Dog") == 1

    dist = DiscreteDistribution({"Dog": 2, "Cat": 3})
    assert all(compare(dist.keys_as_list(), ["Dog", "Cat"]))
    assert dist.rvs.size == 1
    assert dist["Dog"] == 2
    assert dist["Cat"] == 3
    assert dist["Dolphin"] == 0
    assert all(compare(dist.frequencies(normalised=True), [2 / 5, 3 / 5]))
    assert all(compare(dist.frequencies(normalised=False), [2, 3]))
    assert dist.prob("Dog") == 2 / 5
    assert dist.prob(X1="Dog") == 2 / 5
    assert dist.prob("Cat") == 3 / 5
    assert dist.prob(X1="Cat") == 3 / 5
    assert dist.prob("Dolphin") == 0
    assert dist.prob(X1="Dolphin") == 0

    dist = DiscreteDistribution({"Dog": 2, "Cat": 3, "Dolphin": 4})
    assert all(compare(dist.keys_as_list(), ["Dog", "Cat", "Dolphin"]))
    assert dist.rvs.size == 1
    assert dist["Dog"] == 2
    assert dist["Cat"] == 3
    assert dist["Dolphin"] == 4
    assert dist["Tiger"] == 0
    assert all(compare(dist.frequencies(normalised=True), [2 / 9, 3 / 9, 4 / 9]))
    assert all(compare(dist.frequencies(normalised=False), [2, 3, 4]))
    assert dist.prob("Dog") == 2 / 9
    assert dist.prob(X1="Dog") == 2 / 9
    assert dist.prob("Cat") == 3 / 9
    assert dist.prob(X1="Cat") == 3 / 9
    assert dist.prob("Dolphin") == 4 / 9
    assert dist.prob(X1="Dolphin") == 4 / 9
    assert dist.prob("Tiger") == 0
    assert dist.prob(X1="Tiger") == 0


def test_two_levels_discrete_distribution():
    dist = DiscreteDistribution({("A", "y"): 2})
    both_levels = zip(dist.levels(), [["A"], ["y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "y")] == 2
    assert dist[("A", "x")] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob("A", "y") == 1
    assert dist.prob("A", X2="y") == 1
    assert dist.prob("y", X1="A") == 1
    assert dist.prob(X1="A", X2="y") == 1

    dist = DiscreteDistribution({("A", "x"): 2, ("A", "y"): 2})
    both_levels = zip(dist.levels(), [["A"], ["x", "y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "y")] == 2
    assert dist[("A", "x")] == 2
    assert all(compare(dist.frequencies(normalised=True), [2 / 4, 2 / 4]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob("A", "y") == 0.5
    assert dist.prob("A", X2="y") == 0.5
    assert dist.prob("y", X1="A") == 0.5
    assert dist.prob(X1="A", X2="y") == 0.5
    assert dist.prob("A", "x") == 0.5
    assert dist.prob("A", X2="x") == 0.5
    assert dist.prob("x", X1="A") == 0.5
    assert dist.prob(X1="A", X2="x") == 0.5

    dist = DiscreteDistribution({("A", "x"): 2, ("B", "y"): 2})
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "x")] == 2
    assert dist[("B", "y")] == 2
    assert all(compare(dist.frequencies(normalised=True), [2 / 4, 2 / 4]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob("A", "x") == 0.5
    assert dist.prob("A", X2="x") == 0.5
    assert dist.prob("x", X1="A") == 0.5
    assert dist.prob(X1="A", X2="x") == 0.5
    assert dist.prob("B", "y") == 0.5
    assert dist.prob("B", X2="y") == 0.5
    assert dist.prob("y", X1="B") == 0.5
    assert dist.prob(X1="B", X2="y") == 0.5

    dist = DiscreteDistribution({("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3})
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 0
    assert all(compare(dist.frequencies(normalised=True), [1 / 6, 2 / 6, 3 / 6]))
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3]))
    assert dist.prob(X1="A", X2="x") == 1 / 6
    assert dist.prob(X1="A", X2="y") == 2 / 6
    assert dist.prob(X1="B", X2="x") == 3 / 6
    assert dist.prob(X1="B", X2="y") == 0

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4}
    )
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 4
    assert all(compare(dist.frequencies(normalised=True), [0.1, 0.2, 0.3, 0.4]))
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3, 4]))
    assert dist.prob(X1="A", X2="x") == 1 / 10
    assert dist.prob(X1="A", X2="y") == 2 / 10
    assert dist.prob(X1="B", X2="x") == 3 / 10
    assert dist.prob(X1="B", X2="y") == 4 / 10

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4, ("C", "y"): 5}
    )
    both_levels = zip(dist.levels(), [["A", "B", "C"], ["x", "y"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 4
    assert dist[("C", "x")] == 0
    assert dist[("C", "y")] == 5
    assert all(
        compare(
            dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
        )
    )
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3, 4, 5]))
    assert dist.prob(X1="A", X2="x") == 1 / 15
    assert dist.prob(X1="A", X2="y") == 2 / 15
    assert dist.prob(X1="B", X2="x") == 3 / 15
    assert dist.prob(X1="B", X2="y") == 4 / 15
    assert dist.prob(X1="C", X2="x") == 0
    assert dist.prob(X1="C", X2="y") == 5 / 15

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4, ("C", "z"): 5}
    )
    both_levels = zip(dist.levels(), [["A", "B", "C"], ["x", "y", "z"]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 2
    assert dist.prob(X1="A", X2="x") == 1 / 15
    assert dist.prob(X1="A", X2="y") == 2 / 15
    assert dist.prob(X1="B", X2="x") == 3 / 15
    assert dist.prob(X1="B", X2="y") == 4 / 15
    assert dist.prob(X1="C", X2="z") == 5 / 15
    assert dist.prob(X1="A", X2="z") == 0
    assert dist.prob(X1="B", X2="z") == 0
    assert dist.prob(X1="C", X2="x") == 0
    assert dist.prob(X1="C", X2="y") == 0


def test_three_levels_discrete_distribution():
    dist = DiscreteDistribution({("A", "y", 1): 2})
    both_levels = zip(dist.levels(), [["A"], ["y"], [1]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "y", 1)] == 2
    assert dist[("A", "x", 2)] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob(X1="A", X2="y", X3=1) == 1
    assert dist.prob(X1="A", X2="y", X3=2) == 0

    dist = DiscreteDistribution({("A", "x", 1): 2, ("A", "y", 1): 2})
    both_levels = zip(dist.levels(), [["A"], ["x", "y"], [1]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 2
    assert dist[("A", "y", 1)] == 2
    assert all(compare(dist.frequencies(normalised=True), [0.5, 0.5]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob(X1="A", X2="x", X3=1) == 0.5
    assert dist.prob(X1="A", X2="y", X3=1) == 0.5
    assert dist.prob(X1="A", X2="y", X3=2) == 0

    dist = DiscreteDistribution({("A", "x", 1): 2, ("B", "y", 2): 2})
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"], [1, 2]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 2
    assert dist[("B", "y", 2)] == 2
    assert dist[("B", "y", 3)] == 0
    assert all(compare(dist.frequencies(normalised=True), [0.5, 0.5]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob(X1="A", X2="x", X3=1) == 0.5
    assert dist.prob(X1="B", X2="y", X3=2) == 0.5

    dist = DiscreteDistribution({("A", "x", 1): 1, ("A", "y", 2): 2, ("B", "x", 1): 3})
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"], [1, 2]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist["B"] == 0
    assert all(compare(dist.frequencies(normalised=True), [1 / 6, 2 / 6, 3 / 6]))
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3]))
    assert dist.prob(X1="A", X2="x", X3=1) == 1 / 6
    assert dist.prob(X1="A", X2="y", X3=2) == 2 / 6
    assert dist.prob(X1="B", X2="x", X3=1) == 3 / 6
    assert dist.prob(X1="B", X2="y", X3=2) == 0

    dist = DiscreteDistribution(
        {("A", "x", 1): 1, ("A", "y", 2): 2, ("B", "x", 1): 3, ("B", "y", 2): 4}
    )
    both_levels = zip(dist.levels(), [["A", "B"], ["x", "y"], [1, 2]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist[("B", "y", 2)] == 4
    assert all(compare(dist.frequencies(normalised=True), [0.1, 0.2, 0.3, 0.4]))
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3, 4]))
    assert dist.prob(X1="A", X2="x", X3=1) == 1 / 10
    assert dist.prob(X1="A", X2="y", X3=2) == 2 / 10
    assert dist.prob(X1="B", X2="x", X3=1) == 3 / 10
    assert dist.prob(X1="B", X2="y", X3=2) == 4 / 10

    dist = DiscreteDistribution(
        {
            ("A", "x", 1): 1,
            ("A", "y", 2): 2,
            ("B", "x", 1): 3,
            ("B", "y", 2): 4,
            ("C", "y", 3): 5,
        }
    )
    both_levels = zip(dist.levels(), [["A", "B", "C"], ["x", "y"], [1, 2, 3]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist[("B", "y", 2)] == 4
    assert dist[("C", "y", 3)] == 5
    assert all(
        compare(
            dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
        )
    )
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3, 4, 5]))
    assert dist.prob(X1="A", X2="x", X3=1) == 1 / 15
    assert dist.prob(X1="A", X2="y", X3=2) == 2 / 15
    assert dist.prob(X1="B", X2="x", X3=1) == 3 / 15
    assert dist.prob(X1="B", X2="y", X3=2) == 4 / 15
    assert dist.prob(X1="C", X2="y", X3=3) == 5 / 15

    dist = DiscreteDistribution(
        {
            ("A", "x", 1): 1,
            ("A", "y", 2): 2,
            ("B", "x", 3): 3,
            ("B", "y", 3): 4,
            ("C", "z", 4): 5,
        }
    )
    both_levels = zip(dist.levels(), [["A", "B", "C"], ["x", "y", "z"], [1, 2, 3, 4]])
    for levels_1, levels_2 in both_levels:
        assert all(compare(levels_1, levels_2))
    assert dist.rvs.size == 3
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 3)] == 3
    assert dist[("B", "y", 3)] == 4
    assert dist[("C", "z", 4)] == 5
    assert all(
        compare(
            dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
        )
    )
    assert all(compare(dist.frequencies(normalised=False), [1, 2, 3, 4, 5]))
    assert dist.prob(X1="A", X2="x", X3=1) == 1 / 15
    assert dist.prob(X1="A", X2="y", X3=2) == 2 / 15
    assert dist.prob(X1="B", X2="x", X3=3) == 3 / 15
    assert dist.prob(X1="B", X2="y", X3=3) == 4 / 15
    assert dist.prob(X1="C", X2="z", X3=4) == 5 / 15


def test_levels_is_numeric_discrete_distribution():
    samples = {
        ("a", "x", 1, 33, 1.5): 1,
        ("a", "x", 2, 33, 1.5): 2,
        ("a", "x", 1, 44, 1.5): 3,
        ("a", "x", 2, 44, 1.5): 4,
        ("a", "y", 1, 33, 1.5): 5,
        ("a", "y", 2, 33, 1.5): 6,
        ("a", "y", 1, 44, 2.5): 7,
        ("a", "y", 2, 44, 2.5): 8,
        ("b", "x", 1, 33, 2.5): 9,
        ("b", "x", 2, 33, 2.5): 10,
        ("b", "x", 1, 44, 2.5): 11,
        ("b", "x", 2, 44, 2.5): 12,
        ("b", "y", 1, 33, 3.5): 13,
        ("b", "y", 2, 33, 3.5): 14,
        ("b", "y", 1, 44, 3.5): 15,
        ("b", "y", 2, 44, 3.5): 16,
    }

    dist = DiscreteDistribution(samples)
    # by index
    assert not dist.rvs[0].is_numeric
    assert not dist.rvs[1].is_numeric
    assert dist.rvs[2].is_numeric
    assert dist.rvs[3].is_numeric
    assert dist.rvs[4].is_numeric
    # by name
    assert not dist.rvs["X1"].is_numeric
    assert not dist.rvs["X2"].is_numeric
    assert dist.rvs["X3"].is_numeric
    assert dist.rvs["X4"].is_numeric
    assert dist.rvs["X5"].is_numeric


def test_avg_discrete_distribution():
    samples = {
        (1, 1, 1): 1,
        (1, 1, 2): 1,
        (1, 1, 3): 1,
        (1, 2, 1): 2,
        (1, 2, 2): 2,
        (1, 2, 3): 2,
        (1, 3, 1): 3,
        (1, 3, 2): 3,
        (1, 3, 3): 3,
    }
    dist = DiscreteDistribution(samples)
    assert all(compare(dist.avg(), [1, (3 + 12 + 27) / 18, 2]))
    assert all(compare(dist.avg(indices=[0, 1, 2]), [1, (3 + 12 + 27) / 18, 2]))
    assert all(compare(dist.avg(indices=[0, 2, 1]), [1, 2, (3 + 12 + 27) / 18]))
    assert all(compare(dist.avg(indices=[0, 1]), [1, (3 + 12 + 27) / 18]))
    assert all(compare(dist.avg(indices=[0, 2]), [1, 2]))
    assert all(compare(dist.avg(indices=[2, 0]), [2, 1]))
    assert all(compare(dist.avg(indices=[1, 2]), [(3 + 12 + 27) / 18, 2]))
    assert dist.avg(indices=[0]) == 1
    assert dist.std(indices=[0]) == 0
    assert dist.avg(indices=[1]) == (3 + 12 + 27) / 18
    assert dist.std(indices=[1]) == approx(0.55555555555556)
    assert dist.avg(indices=[2]) == 2
    assert dist.std(indices=[2]) == approx(0.66666666666667)
