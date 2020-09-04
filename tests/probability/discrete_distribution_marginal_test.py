import pytest
import numpy as np
from numpy.testing import assert_array_equal as assert_arr
from probability.distributions import DiscreteDistribution


def test_marginals_names_exception_discrete_distribution():
    with pytest.raises(TypeError):
        samples = {"a": 3, "b": 4, "c": 5}
        disc_dist = DiscreteDistribution(samples)
        disc_dist.marginal(["X1"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples)
        disc_dist.marginal(["X0"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples)
        disc_dist.marginal(["X3"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples)
        disc_dist2 = disc_dist.marginal(["X1"])
        disc_dist2.marginal(["X1"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples, names=["Y", "Z"])
        disc_dist.marginal(["X1"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples, names=["Y", "Z"])
        disc_dist.marginal(["X1"])

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        disc_dist = DiscreteDistribution(samples, names=["Y", "Z"])
        disc_dist2 = disc_dist.marginal(["Y"])
        disc_dist2.marginal(["Y"])


def test_marginals_names_discrete_distribution():
    samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
    disc_dist = DiscreteDistribution(samples)

    disc_dist2 = disc_dist.marginal(["X1"])
    assert_arr(disc_dist2.names, ["X2"])

    disc_dist2 = disc_dist.marginal(["X2"])
    assert_arr(disc_dist2.names, ["X1"])
    #
    disc_dist = DiscreteDistribution(samples, names=["Y", "Z"])

    disc_dist2 = disc_dist.marginal(["Y"])
    assert_arr(disc_dist2.names, ["Z"])

    disc_dist2 = disc_dist.marginal(["Z"])
    assert_arr(disc_dist2.names, ["Y"])

    # Three levels dist.
    samples = {
        ("a", "x", 1): 4,
        ("a", "x", 2): 4,
        ("a", "y", 1): 6,
        ("a", "y", 2): 6,
        ("b", "x", 1): 8,
        ("b", "x", 2): 8,
        ("b", "y", 1): 10,
        ("b", "y", 2): 10,
    }

    disc_dist = DiscreteDistribution(samples)

    disc_dist2 = disc_dist.marginal(["X1"])
    assert_arr(disc_dist2.names, ["X2", "X3"])

    disc_dist2 = disc_dist.marginal(["X2"])
    assert_arr(disc_dist2.names, ["X1", "X3"])

    disc_dist2 = disc_dist.marginal(["X3"])
    assert_arr(disc_dist2.names, ["X1", "X2"])

    disc_dist2 = disc_dist.marginal(["X1", "X3"])
    assert_arr(disc_dist2.names, ["X2"])

    disc_dist2 = disc_dist.marginal(["X2", "X3"])
    assert_arr(disc_dist2.names, ["X1"])

    #
    disc_dist = DiscreteDistribution(samples, names=["Y", "Z", "W"])

    disc_dist2 = disc_dist.marginal(["Y"])
    assert_arr(disc_dist2.names, ["Z", "W"])

    disc_dist2 = disc_dist.marginal(["Z"])
    assert_arr(disc_dist2.names, ["Y", "W"])

    disc_dist2 = disc_dist.marginal(["W"])
    assert_arr(disc_dist2.names, ["Y", "Z"])

    disc_dist2 = disc_dist.marginal(["Y", "W"])
    assert_arr(disc_dist2.names, ["Z"])

    disc_dist2 = disc_dist.marginal(["Z", "W"])
    assert_arr(disc_dist2.names, ["Y"])


def test_marginals_discrete_distribution():
    # Single RV dist.
    with pytest.raises(TypeError):
        disc_dist = DiscreteDistribution({"A": 2, "B": 3, "C": 4})
        disc_dist.marginal_by_indices([1])

    # Two levels dist.
    samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
    disc_dist = DiscreteDistribution(samples)
    disc_dist2 = disc_dist.marginal_by_indices([0])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), ["x", "y"])
    assert disc_dist2["x"] == 10
    assert disc_dist2["y"] == 10
    assert disc_dist2.probability("x") == 0.5
    assert disc_dist2.probability("y") == 0.5

    disc_dist2 = disc_dist.marginal_by_indices([0])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), ["x", "y"])
    assert disc_dist2["x"] == 10
    assert disc_dist2["y"] == 10
    assert disc_dist2.probability("x") == 0.5
    assert disc_dist2.probability("y") == 0.5

    disc_dist2 = disc_dist.marginal_by_indices([1])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), ["a", "b"])
    assert disc_dist2["a"] == 8
    assert disc_dist2["b"] == 12
    assert disc_dist2.probability("a") == 0.4
    assert disc_dist2.probability("b") == 0.6

    # Three levels dist.
    samples = {
        ("a", "x", 1): 4,
        ("a", "x", 2): 4,
        ("a", "y", 1): 6,
        ("a", "y", 2): 6,
        ("b", "x", 1): 8,
        ("b", "x", 2): 8,
        ("b", "y", 1): 10,
        ("b", "y", 2): 10,
    }
    disc_dist = DiscreteDistribution(samples)
    disc_dist2 = disc_dist.marginal_by_indices([0])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array([("x", 1), ("x", 2), ("y", 1), ("y", 2)], dtype=np.object),
    )
    assert disc_dist2[("x", 1)] == 12
    assert disc_dist2[("x", 2)] == 12
    assert disc_dist2[("y", 1)] == 16
    assert disc_dist2[("y", 2)] == 16
    assert disc_dist2.probability(("x", 1)) == 12 / 56
    assert disc_dist2.probability(("x", 2)) == 12 / 56
    assert disc_dist2.probability(("y", 1)) == 16 / 56
    assert disc_dist2.probability(("y", 2)) == 16 / 56

    disc_dist2 = disc_dist.marginal_by_indices([1])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array([("a", 1), ("a", 2), ("b", 1), ("b", 2)], dtype=np.object),
    )
    assert disc_dist2[("a", 1)] == 10
    assert disc_dist2[("a", 2)] == 10
    assert disc_dist2[("b", 1)] == 18
    assert disc_dist2[("b", 2)] == 18
    assert disc_dist2.probability(("a", 1)) == 10 / 56
    assert disc_dist2.probability(("a", 2)) == 10 / 56
    assert disc_dist2.probability(("b", 1)) == 18 / 56
    assert disc_dist2.probability(("b", 2)) == 18 / 56

    disc_dist2 = disc_dist.marginal_by_indices([2])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array([("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")], dtype=np.object),
    )
    assert disc_dist2[("a", "x")] == 8
    assert disc_dist2[("a", "y")] == 12
    assert disc_dist2[("b", "x")] == 16
    assert disc_dist2[("b", "y")] == 20
    assert disc_dist2.probability(("a", "x")) == 8 / 56
    assert disc_dist2.probability(("a", "y")) == 12 / 56
    assert disc_dist2.probability(("b", "x")) == 16 / 56
    assert disc_dist2.probability(("b", "y")) == 20 / 56

    disc_dist2 = disc_dist.marginal_by_indices([0, 1])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), np.array([1, 2], dtype=np.object))
    assert disc_dist2[1] == 28
    assert disc_dist2[2] == 28
    assert disc_dist2.probability(1) == 28 / 56
    assert disc_dist2.probability(2) == 28 / 56

    disc_dist2 = disc_dist.marginal_by_indices([0, 2])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), np.array(["x", "y"], dtype=np.object))
    assert disc_dist2["x"] == 24
    assert disc_dist2["y"] == 32
    assert disc_dist2.probability("x") == 24 / 56
    assert disc_dist2.probability("y") == 32 / 56

    disc_dist2 = disc_dist.marginal_by_indices([1, 2])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), np.array(["a", "b"], dtype=np.object))
    assert disc_dist2["a"] == 20
    assert disc_dist2["b"] == 36
    assert disc_dist2.probability("a") == 20 / 56
    assert disc_dist2.probability("b") == 36 / 56

    # Four levels dist.
    samples = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    disc_dist = DiscreteDistribution(samples)
    disc_dist2 = disc_dist.marginal_by_indices([2])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array(
            [
                ("a", "x", 33),
                ("a", "x", 44),
                ("a", "y", 33),
                ("a", "y", 44),
                ("b", "x", 33),
                ("b", "x", 44),
                ("b", "y", 33),
                ("b", "y", 44),
            ],
            dtype=np.object,
        ),
    )
    assert disc_dist2[("a", "x", 33)] == 3
    assert disc_dist2[("a", "x", 44)] == 7
    assert disc_dist2[("a", "y", 33)] == 11
    assert disc_dist2[("a", "y", 44)] == 15
    assert disc_dist2[("b", "x", 33)] == 19
    assert disc_dist2[("b", "x", 44)] == 23
    assert disc_dist2[("b", "y", 33)] == 27
    assert disc_dist2[("b", "y", 44)] == 31
    assert disc_dist2.probability(("a", "x", 33)) == 3 / 136
    assert disc_dist2.probability(("a", "x", 44)) == 7 / 136
    assert disc_dist2.probability(("a", "y", 33)) == 11 / 136
    assert disc_dist2.probability(("a", "y", 44)) == 15 / 136
    assert disc_dist2.probability(("b", "x", 33)) == 19 / 136
    assert disc_dist2.probability(("b", "x", 44)) == 23 / 136
    assert disc_dist2.probability(("b", "y", 33)) == 27 / 136
    assert disc_dist2.probability(("b", "y", 44)) == 31 / 136

    disc_dist2 = disc_dist.marginal_by_indices([3])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array(
            [
                ("a", "x", 1),
                ("a", "x", 2),
                ("a", "y", 1),
                ("a", "y", 2),
                ("b", "x", 1),
                ("b", "x", 2),
                ("b", "y", 1),
                ("b", "y", 2),
            ],
            dtype=np.object,
        ),
    )
    assert disc_dist2[("a", "x", 1)] == 4
    assert disc_dist2[("a", "x", 2)] == 6
    assert disc_dist2[("a", "y", 1)] == 12
    assert disc_dist2[("a", "y", 2)] == 14
    assert disc_dist2[("b", "x", 1)] == 20
    assert disc_dist2[("b", "x", 2)] == 22
    assert disc_dist2[("b", "y", 1)] == 28
    assert disc_dist2[("b", "y", 2)] == 30
    assert disc_dist2.probability(("a", "x", 1)) == 4 / 136
    assert disc_dist2.probability(("a", "x", 2)) == 6 / 136
    assert disc_dist2.probability(("a", "y", 1)) == 12 / 136
    assert disc_dist2.probability(("a", "y", 2)) == 14 / 136
    assert disc_dist2.probability(("b", "x", 1)) == 20 / 136
    assert disc_dist2.probability(("b", "x", 2)) == 22 / 136
    assert disc_dist2.probability(("b", "y", 1)) == 28 / 136
    assert disc_dist2.probability(("b", "y", 2)) == 30 / 136

    disc_dist2 = disc_dist.marginal_by_indices([0, 3])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array([("x", 1), ("x", 2), ("y", 1), ("y", 2)], dtype=np.object),
    )
    assert disc_dist2[("x", 1)] == 24
    assert disc_dist2[("x", 2)] == 28
    assert disc_dist2[("y", 1)] == 40
    assert disc_dist2[("y", 2)] == 44
    assert disc_dist2.probability(("x", 1)) == 24 / 136
    assert disc_dist2.probability(("x", 2)) == 28 / 136
    assert disc_dist2.probability(("y", 1)) == 40 / 136
    assert disc_dist2.probability(("y", 2)) == 44 / 136

    disc_dist2 = disc_dist.marginal_by_indices([0, 1, 3])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), np.array([1, 2], dtype=np.object))
    assert disc_dist2[1] == 64
    assert disc_dist2[2] == 72
    assert disc_dist2.probability(1) == 64 / 136
    assert disc_dist2.probability(2) == 72 / 136

    # marginalize two times
    disc_dist2 = disc_dist.marginal_by_indices([0, 3])
    disc_dist3 = disc_dist2.marginal_by_indices([0])
    assert disc_dist3.total == disc_dist.total
    assert_arr(disc_dist3.np_keys(), np.array([1, 2], dtype=np.object))
    assert disc_dist3[1] == 64
    assert disc_dist3[2] == 72
    assert disc_dist3.probability(1) == 64 / 136
    assert disc_dist3.probability(2) == 72 / 136

    # marginalize three times
    disc_dist2 = disc_dist.marginal_by_indices([3])
    disc_dist3 = disc_dist2.marginal_by_indices([2])
    disc_dist4 = disc_dist3.marginal_by_indices([1])
    assert disc_dist4.total == disc_dist.total
    assert_arr(disc_dist4.np_keys(), np.array(["a", "b"], dtype=np.object))
    assert disc_dist4["a"] == 36
    assert disc_dist4["b"] == 100
    assert disc_dist4.probability("a") == 36 / 136
    assert disc_dist4.probability("b") == 100 / 136


def test_marginal_by_name_discrete_distribution():
    # Four levels dist.
    samples = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    disc_dist = DiscreteDistribution(samples)
    disc_dist2 = disc_dist.marginal(["X3"])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array(
            [
                ("a", "x", 33),
                ("a", "x", 44),
                ("a", "y", 33),
                ("a", "y", 44),
                ("b", "x", 33),
                ("b", "x", 44),
                ("b", "y", 33),
                ("b", "y", 44),
            ],
            dtype=np.object,
        ),
    )
    assert disc_dist2[("a", "x", 33)] == 3
    assert disc_dist2[("a", "x", 44)] == 7
    assert disc_dist2[("a", "y", 33)] == 11
    assert disc_dist2[("a", "y", 44)] == 15
    assert disc_dist2[("b", "x", 33)] == 19
    assert disc_dist2[("b", "x", 44)] == 23
    assert disc_dist2[("b", "y", 33)] == 27
    assert disc_dist2[("b", "y", 44)] == 31
    assert disc_dist2.probability(("a", "x", 33)) == 3 / 136
    assert disc_dist2.probability(("a", "x", 44)) == 7 / 136
    assert disc_dist2.probability(("a", "y", 33)) == 11 / 136
    assert disc_dist2.probability(("a", "y", 44)) == 15 / 136
    assert disc_dist2.probability(("b", "x", 33)) == 19 / 136
    assert disc_dist2.probability(("b", "x", 44)) == 23 / 136
    assert disc_dist2.probability(("b", "y", 33)) == 27 / 136
    assert disc_dist2.probability(("b", "y", 44)) == 31 / 136

    disc_dist2 = disc_dist.marginal(["X4"])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array(
            [
                ("a", "x", 1),
                ("a", "x", 2),
                ("a", "y", 1),
                ("a", "y", 2),
                ("b", "x", 1),
                ("b", "x", 2),
                ("b", "y", 1),
                ("b", "y", 2),
            ],
            dtype=np.object,
        ),
    )
    assert disc_dist2[("a", "x", 1)] == 4
    assert disc_dist2[("a", "x", 2)] == 6
    assert disc_dist2[("a", "y", 1)] == 12
    assert disc_dist2[("a", "y", 2)] == 14
    assert disc_dist2[("b", "x", 1)] == 20
    assert disc_dist2[("b", "x", 2)] == 22
    assert disc_dist2[("b", "y", 1)] == 28
    assert disc_dist2[("b", "y", 2)] == 30
    assert disc_dist2.probability(("a", "x", 1)) == 4 / 136
    assert disc_dist2.probability(("a", "x", 2)) == 6 / 136
    assert disc_dist2.probability(("a", "y", 1)) == 12 / 136
    assert disc_dist2.probability(("a", "y", 2)) == 14 / 136
    assert disc_dist2.probability(("b", "x", 1)) == 20 / 136
    assert disc_dist2.probability(("b", "x", 2)) == 22 / 136
    assert disc_dist2.probability(("b", "y", 1)) == 28 / 136
    assert disc_dist2.probability(("b", "y", 2)) == 30 / 136

    disc_dist2 = disc_dist.marginal(["X1", "X4"])
    assert disc_dist2.total == disc_dist.total
    assert_arr(
        disc_dist2.np_keys(),
        np.array([("x", 1), ("x", 2), ("y", 1), ("y", 2)], dtype=np.object),
    )
    assert disc_dist2[("x", 1)] == 24
    assert disc_dist2[("x", 2)] == 28
    assert disc_dist2[("y", 1)] == 40
    assert disc_dist2[("y", 2)] == 44
    assert disc_dist2.probability(("x", 1)) == 24 / 136
    assert disc_dist2.probability(("x", 2)) == 28 / 136
    assert disc_dist2.probability(("y", 1)) == 40 / 136
    assert disc_dist2.probability(("y", 2)) == 44 / 136

    disc_dist2 = disc_dist.marginal(["X1", "X2", "X4"])
    assert disc_dist2.total == disc_dist.total
    assert_arr(disc_dist2.np_keys(), np.array([1, 2], dtype=np.object))
    assert disc_dist2[1] == 64
    assert disc_dist2[2] == 72
    assert disc_dist2.probability(1) == 64 / 136
    assert disc_dist2.probability(2) == 72 / 136

    # marginalize two times
    disc_dist2 = disc_dist.marginal(["X1", "X4"])
    disc_dist3 = disc_dist2.marginal(["X2"])
    assert disc_dist3.total == disc_dist.total
    assert_arr(disc_dist3.np_keys(), np.array([1, 2], dtype=np.object))
    assert disc_dist3[1] == 64
    assert disc_dist3[2] == 72
    assert disc_dist3.probability(1) == 64 / 136
    assert disc_dist3.probability(2) == 72 / 136

    # marginalize three times
    disc_dist2 = disc_dist.marginal(["X4"])
    disc_dist3 = disc_dist2.marginal(["X3"])
    disc_dist4 = disc_dist3.marginal(["X2"])
    assert disc_dist4.total == disc_dist.total
    assert_arr(disc_dist4.np_keys(), np.array(["a", "b"], dtype=np.object))
    assert disc_dist4["a"] == 36
    assert disc_dist4["b"] == 100
    assert disc_dist4.probability("a") == 36 / 136
    assert disc_dist4.probability("b") == 100 / 136


def test_marginals_operator_discrete_distribution():
    # Four levels dist.
    samples = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    disc_dist = DiscreteDistribution(samples)
    assert (disc_dist << ["X2"]).total == disc_dist.total
    assert (disc_dist << ["X2", "X3"]).total == disc_dist.total
    assert (disc_dist << ["X2", "X3", "X4"]).total == disc_dist.total

    assert_arr((disc_dist << ["X1", "X2", "X4"]).np_keys(), [1, 2])
    assert_arr((disc_dist << ["X1", "X2", "X3"]).np_keys(), [33, 44])
    assert_arr((disc_dist << ["X2", "X3", "X4"]).np_keys(), ["a", "b"])
    assert_arr(
        (disc_dist << ["X2", "X3"]).np_keys(),
        np.array([("a", 33), ("a", 44), ("b", 33), ("b", 44)], dtype=np.object),
    )

    disc_dist = DiscreteDistribution(samples, names=["Age", "Sex", "Education", "City"])
    assert (disc_dist << ["Age"]).total == disc_dist.total
    assert (disc_dist << ["Sex", "Education"]).total == disc_dist.total
    assert (disc_dist << ["Sex", "Education", "City"]).total == disc_dist.total

    assert_arr((disc_dist << ["Age", "Sex", "City"]).np_keys(), [1, 2])
    assert_arr((disc_dist << ["Age", "Sex", "Education"]).np_keys(), [33, 44])
    assert_arr((disc_dist << ["Sex", "Education", "City"]).np_keys(), ["a", "b"])
    assert_arr(
        (disc_dist << ["Sex", "Education"]).np_keys(),
        np.array([("a", 33), ("a", 44), ("b", 33), ("b", 44)], dtype=np.object),
    )