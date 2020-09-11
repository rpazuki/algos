from probability.distributions import DiscreteDistribution
from tests.helpers import compare


def test_reduce_by_name_discrete_distribution():
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
    reduced_dist = disc_dist.reduce(X2="y")
    assert reduced_dist.rvs.size == 3
    assert all(compare(reduced_dist.rvs.names, ["X1", "X3", "X4"]))
    assert reduced_dist[("a", 1, 33)] == 5
    assert reduced_dist[("b", 2, 44)] == 16
    assert reduced_dist.frequency(("a", 1, 33)) == 5
    assert reduced_dist.frequency(("b", 2, 44)) == 16
    assert reduced_dist.probability(("a", 1, 33)) == 5 / 84
    assert reduced_dist.probability(("b", 2, 44)) == 16 / 84

    reduced_dist = disc_dist.reduce(X2="y", X3=1)
    assert reduced_dist.rvs.size == 2
    assert all(compare(reduced_dist.rvs.names, ["X1", "X4"]))
    assert reduced_dist[("a", 33)] == 5
    assert reduced_dist[("b", 44)] == 15
    assert reduced_dist.frequency(("a", 33)) == 5
    assert reduced_dist.frequency(("b", 44)) == 15
    assert reduced_dist.probability(("a", 33)) == 5 / 40
    assert reduced_dist.probability(("b", 44)) == 15 / 40

    reduced_dist = disc_dist.reduce(X1="b", X3=1, X4=44)
    assert reduced_dist.rvs.size == 1
    assert all(compare(reduced_dist.rvs.names, ["X2"]))
    assert reduced_dist["x"] == 11
    assert reduced_dist["y"] == 15
    assert reduced_dist.frequency("x") == 11
    assert reduced_dist.frequency("y") == 15
    assert reduced_dist.probability("x") == 11 / 26
    assert reduced_dist.probability("y") == 15 / 26

    disc_dist = DiscreteDistribution(samples, names=["Y", "Z", "W", "X"])

    reduced_dist = disc_dist.reduce(Z="y")
    assert reduced_dist.rvs.size == 3
    assert all(compare(reduced_dist.rvs.names, ["Y", "W", "X"]))
    assert reduced_dist[("a", 1, 33)] == 5
    assert reduced_dist[("b", 2, 44)] == 16
    assert reduced_dist.frequency(("a", 1, 33)) == 5
    assert reduced_dist.frequency(("b", 2, 44)) == 16
    assert reduced_dist.probability(("a", 1, 33)) == 5 / 84
    assert reduced_dist.probability(("b", 2, 44)) == 16 / 84

    reduced_dist = disc_dist.reduce(Z="y", W=1)
    assert reduced_dist.rvs.size == 2
    assert all(compare(reduced_dist.rvs.names, ["Y", "X"]))
    assert reduced_dist[("a", 33)] == 5
    assert reduced_dist[("b", 44)] == 15
    assert reduced_dist.frequency(("a", 33)) == 5
    assert reduced_dist.frequency(("b", 44)) == 15
    assert reduced_dist.probability(("a", 33)) == 5 / 40
    assert reduced_dist.probability(("b", 44)) == 15 / 40

    reduced_dist = disc_dist.reduce(Y="b", W=1, X=44)
    assert reduced_dist.rvs.size == 1
    assert all(compare(reduced_dist.rvs.names, ["Z"]))
    assert reduced_dist["x"] == 11
    assert reduced_dist["y"] == 15
    assert reduced_dist.frequency("x") == 11
    assert reduced_dist.frequency("y") == 15
    assert reduced_dist.probability("x") == 11 / 26
    assert reduced_dist.probability("y") == 15 / 26
