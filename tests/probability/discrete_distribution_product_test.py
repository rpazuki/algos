import pytest
from pytest import approx
import numpy as np
from probability.distributions import DiscreteDistribution
from probability.distributions import FrequencyTable
from tests.helpers import compare

sample_1 = {
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
    # ("b", "y", 2, 33): 14,
    ("b", "y", 1, 44): 15,
    ("b", "y", 2, 44): 16,
}

sample_2 = {
    (1, "high", "under", "x"): 1,
    (1, "high", "normal", "x"): 2,
    (1, "high", "over", "x"): 3,
    (1, "high", "obese", "x"): 4,
    (1, "low", "under", "x"): 5,
    (1, "low", "normal", "x"): 6,
    (1, "low", "over", "x"): 7,
    (1, "low", "obese", "x"): 8,
    (2, "high", "under", "x"): 9,
    (2, "high", "normal", "x"): 10,
    (2, "high", "over", "x"): 11,
    (2, "high", "obese", "x"): 12,
    (2, "low", "under", "x"): 13,
    (2, "low", "normal", "x"): 14,
    (2, "low", "over", "x"): 15,
    (2, "low", "obese", "x"): 16,
    (1, "high", "under", "y"): 17,
    (1, "high", "normal", "y"): 18,
    (1, "high", "over", "y"): 19,
    (1, "high", "obese", "y"): 20,
    (1, "low", "under", "y"): 21,
    (1, "low", "normal", "y"): 22,
    (1, "low", "over", "y"): 23,
    (1, "low", "obese", "y"): 24,
    (2, "high", "under", "y"): 25,
    (2, "high", "normal", "y"): 26,
    # (2, "high", "over", "y"): 27,
    # (2, "high", "obese", "y"): 28,
    # (2, "low", "under", "y"): 29,
    # (2, "low", "normal", "y"): 30,
    # (2, "low", "over", "y"): 31,
    # (2, "low", "obese", "y"): 32,
}


def test_product_exceptions_discrete_distribution():
    dist1 = DiscreteDistribution(sample_1)
    with pytest.raises(ValueError):
        dist1.product(2)


def test_product_with_one_common_var_discrete_distribution():

    dist1 = DiscreteDistribution(sample_1, names=["X1", "X2", "X3", "X4"])
    dist2 = DiscreteDistribution(sample_2, names=["X3", "X5", "X6", "X7"])

    dist3 = dist1 * dist2
    assert all(compare(dist3.names, ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]))
    assert dist3.total == (36 + 164) * 64 + (100 + 51) * 58

    # check probabilites
    assert dist3.frequency(("a", "x", 1, 33, "high", "normal", "x")) == 2
    assert dist3.probability(("a", "x", 1, 33, "high", "normal", "x")) == 2 / 21558
    assert dist3[("a", "x", 1, 33, "high", "normal", "x")] == 2
    # check the case that the right does not have the common
    assert dist3.frequency(("b", "y", 2, 44, "high", "over", "y")) == 0
    assert dist3.probability(("b", "y", 2, 44, "high", "over", "y")) == 0
    assert dist3[("b", "y", 2, 44, "high", "over", "y")] == 0
    # check the case that the left does not have the common
    assert dist3.frequency(("b", "y", 2, 33, "high", "normal", "y")) == 0
    assert dist3.probability(("b", "y", 2, 33, "high", "normal", "y")) == 0
    assert dist3[("b", "y", 2, 33, "high", "normal", "y")] == 0


def test_product_with_two_common_vars_discrete_distribution():
    dist1 = DiscreteDistribution(sample_1, names=["X1", "X2", "X3", "X4"])
    dist2 = DiscreteDistribution(sample_2, names=["X3", "X5", "X6", "X2"])

    dist3 = dist1 * dist2
    assert all(compare(dist3.names, ["X1", "X2", "X3", "X4", "X5", "X6"]))
    assert dist3.total == 24 * 36 + 28 * 100 + 40 * 164 + 30 * 51

    # check probabilites
    assert dist3.frequency(("a", "y", 2, 33, "high", "under")) == 6 * 25
    assert dist3.probability(("a", "y", 2, 33, "high", "under")) == 150 / 11754
    assert dist3[("a", "y", 2, 33, "high", "under")] == 150
    # check the case that the right does not have the common
    assert dist3.frequency(("a", "y", 2, 33, "low", "under")) == 0
    assert dist3.probability(("a", "y", 2, 33, "low", "under")) == 0
    assert dist3[("a", "y", 2, 33, "low", "under")] == 0
    # check the case that the left does not have the common
    assert dist3.frequency(("b", "y", 2, 33, "high", "under")) == 0
    assert dist3.probability(("b", "y", 2, 33, "high", "under")) == 0
    assert dist3[("b", "y", 2, 33, "high", "under")] == 0


def test_product_with_no_common_vars_discrete_distribution():

    dist1 = DiscreteDistribution(sample_1, names=["X1", "X2", "X3", "X4"])
    dist2 = DiscreteDistribution(sample_2, names=["Y1", "Y2", "Y3", "Y4"])

    dist3 = dist1 * dist2
    assert all(compare(dist3.names, ["X1", "X2", "X3", "X4", "Y1", "Y2", "Y3", "Y4"]))
    assert dist3.total == (dist1.total * dist2.total)

    # check probabilites
    assert dist3.frequency(("a", "x", 1, 33, 2, "high", "under", "x")) == 9
    assert dist3.probability(("a", "x", 1, 33, 2, "high", "normal", "x")) == 10 / 42822
    assert dist3[("a", "x", 1, 33, 2, "high", "normal", "x")] == 10

    assert dist3.frequency(("b", "x", 1, 44, 1, "low", "over", "y")) == 253
    assert dist3.probability(("b", "x", 1, 44, 1, "low", "over", "y")) == 253 / 42822
    assert dist3[("b", "x", 1, 44, 1, "low", "over", "y")] == 253


def test_product_with_frequency_table_discrete_distribution():
    freq_table1 = FrequencyTable({"A": 3, "B": 4, "C": 7}, name="Y1")
    dist1 = DiscreteDistribution(sample_1, names=["X1", "X2", "X3", "X4"])

    # without common names
    dist3 = dist1 * freq_table1
    assert all(compare(dist3.names, ["X1", "X2", "X3", "X4", "Y1"]))
    assert dist3.total == (dist1.total * freq_table1.total)

    # check probabilites
    assert dist3.frequency(("a", "y", 2, 33, "B")) == 6 * 4
    assert dist3.probability(("a", "y", 2, 33, "B")) == 24 / 1708
    assert dist3[("a", "y", 2, 33, "B")] == 24

    dist3 = freq_table1 * dist1
    assert all(compare(dist3.names, ["Y1", "X1", "X2", "X3", "X4"]))
    assert dist3.total == (dist1.total * freq_table1.total)

    # check probabilites
    assert dist3.frequency(("B", "a", "y", 2, 33)) == 6 * 4
    assert dist3.probability(("B", "a", "y", 2, 33)) == 24 / 1708
    assert dist3[("B", "a", "y", 2, 33)] == 24

    # with common names
    freq_table1 = FrequencyTable({"x": 3, "y": 4}, name="X2")
    dist1 = DiscreteDistribution(sample_1, names=["X1", "X2", "X3", "X4"])

    dist3 = dist1 * freq_table1
    assert all(compare(dist3.names, ["X1", "X2", "X3", "X4"]))
    assert dist3.total == 52 * 3 + 70 * 4

    dist3 = freq_table1 * dist1
    assert all(compare(dist3.names, ["X2", "X1", "X3", "X4"]))
    assert dist3.total == 52 * 3 + 70 * 4


def test_statistical_independence_frequency_table():
    # P(x,y,z) = P(x)P(y)P(z)
    # to check that, first, create a joint dist. by product
    # then marginalis and multiply again. The final must be equal
    # the joint
    # Note: the multi-variable distributions must be statistically
    #       independent
    s_1 = {
        ("x", 1): 1 * 6,
        ("x", 2): 1 * 4,
        ("y", 1): 9 * 6,
        ("y", 2): 9 * 4,
    }

    dist1 = DiscreteDistribution(s_1, names=["X1", "X2"])

    s_2 = {
        (1, "high", "under", "x"): 4 * 3 * 1 * 1,
        (1, "high", "normal", "x"): 4 * 3 * 2 * 1,
        (1, "high", "over", "x"): 4 * 3 * 3 * 1,
        (1, "high", "obese", "x"): 4 * 3 * 4 * 1,
        (1, "low", "under", "x"): 4 * 2 * 1 * 1,
        (1, "low", "normal", "x"): 4 * 2 * 2 * 1,
        (1, "low", "over", "x"): 4 * 2 * 3 * 1,
        (1, "low", "obese", "x"): 4 * 2 * 4 * 1,
        (2, "high", "under", "x"): 2 * 3 * 1 * 1,
        (2, "high", "normal", "x"): 2 * 3 * 2 * 1,
        (2, "high", "over", "x"): 2 * 3 * 3 * 1,
        (2, "high", "obese", "x"): 2 * 3 * 4 * 1,
        (2, "low", "under", "x"): 2 * 2 * 1 * 1,
        (2, "low", "normal", "x"): 2 * 2 * 2 * 1,
        (2, "low", "over", "x"): 2 * 2 * 3 * 1,
        (2, "low", "obese", "x"): 2 * 2 * 4 * 1,
        (1, "high", "under", "y"): 4 * 3 * 1 * 3,
        (1, "high", "normal", "y"): 4 * 3 * 2 * 3,
        (1, "high", "over", "y"): 4 * 3 * 3 * 3,
        (1, "high", "obese", "y"): 4 * 3 * 4 * 3,
        (1, "low", "under", "y"): 4 * 2 * 1 * 3,
        (1, "low", "normal", "y"): 4 * 2 * 2 * 3,
        (1, "low", "over", "y"): 4 * 2 * 3 * 3,
        (1, "low", "obese", "y"): 4 * 2 * 4 * 3,
        (2, "high", "under", "y"): 2 * 3 * 1 * 3,
        (2, "high", "normal", "y"): 2 * 3 * 2 * 3,
        (2, "high", "over", "y"): 2 * 3 * 3 * 3,
        (2, "high", "obese", "y"): 2 * 3 * 4 * 3,
        (2, "low", "under", "y"): 2 * 2 * 1 * 3,
        (2, "low", "normal", "y"): 2 * 2 * 2 * 3,
        (2, "low", "over", "y"): 2 * 2 * 3 * 3,
        (2, "low", "obese", "y"): 2 * 2 * 4 * 3,
    }
    dist2 = DiscreteDistribution(s_2, names=["Y1", "Y2", "Y3", "Y4"])
    freq_table3 = FrequencyTable({11: 2, 22: 4, 33: 3}, name="Z")

    joint_dist = dist1 * dist2 * freq_table3

    marginals = []
    for name in joint_dist.names:
        names_except_one = list(set(joint_dist.names) - {name})
        marginal = joint_dist.marginal(*names_except_one)
        marginals.append(marginal)

    joint_dist2 = np.product(marginals)

    for k1 in joint_dist:
        assert joint_dist.probability(k1) == joint_dist2.probability(k1)

    # Test by normalising the distributions
    dist1.normalise()
    dist2.normalise()
    freq_table3.normalise()

    joint_dist = dist1 * dist2 * freq_table3

    marginals = []
    for name in joint_dist.names:
        names_except_one = list(set(joint_dist.names) - {name})
        marginal = joint_dist.marginal(*names_except_one)
        marginals.append(marginal)

    joint_dist2 = np.product(marginals)

    for k1 in joint_dist:
        assert joint_dist.probability(k1) == approx(
            joint_dist2.probability(k1), abs=1e-16
        )
        assert joint_dist[k1] == approx(joint_dist2[k1], abs=1e-16)
