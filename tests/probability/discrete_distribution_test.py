import pytest
from numpy.testing import assert_array_equal as assert_arr
from probability.distributions import DiscreteDistribution


def test_invalid_samples_discrete_distribution():

    # mismatch length
    with pytest.raises(ValueError):
        DiscreteDistribution({("A", "b", "x"): 2, ("B", "b"): 3})

    with pytest.raises(ValueError):
        DiscreteDistribution({("A", "b", "x"): 2, ("B", "b"): 3, ("A", "b", "y"): 2})

    with pytest.raises(ValueError):
        DiscreteDistribution(
            {("A", "b", "x"): 2, ("B", "b"): 3, ("A", "b", "y"): 2, ("A", "b", "z"): 5}
        )

    with pytest.raises(ValueError):
        DiscreteDistribution(
            {("b", "x"): 2, ("B", "b"): 3, ("A", "b", "y"): 2, ("A", "b", "z"): 5}
        )


def test_one_levels_discrete_distribution():
    dist = DiscreteDistribution({"Dog": 2})
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog"}
    assert dist["Dog"] == 2
    assert dist["Cat"] == 0
    assert_arr(dist.frequencies(normalised=True), [1])
    assert_arr(dist.frequencies(normalised=False), [2])

    dist = DiscreteDistribution({"Dog": 2, "Cat": 3})
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog", "Cat"}
    assert dist["Dog"] == 2
    assert dist["Cat"] == 3
    assert dist["Dolphin"] == 0
    assert_arr(dist.frequencies(normalised=True), [2 / 5, 3 / 5])
    assert_arr(dist.frequencies(normalised=False), [2, 3])

    dist = DiscreteDistribution({"Dog": 2, "Cat": 3, "Dolphin": 4})
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog", "Cat", "Dolphin"}
    assert dist["Dog"] == 2
    assert dist["Cat"] == 3
    assert dist["Dolphin"] == 4
    assert dist["Tiger"] == 0
    assert_arr(dist.frequencies(normalised=True), [2 / 9, 3 / 9, 4 / 9])
    assert_arr(dist.frequencies(normalised=False), [2, 3, 4])


def test_two_levels_discrete_distribution():
    dist = DiscreteDistribution({("A", "y"): 2})
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y"}
    assert dist[("A", "y")] == 2
    assert dist[("A", "x")] == 0
    assert_arr(dist.frequencies(normalised=True), [1])
    assert_arr(dist.frequencies(normalised=False), [2])

    dist = DiscreteDistribution({("A", "x"): 2, ("A", "y"): 2})
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist[("A", "y")] == 2
    assert dist[("A", "x")] == 2
    assert_arr(dist.frequencies(normalised=True), [2 / 4, 2 / 4])
    assert_arr(dist.frequencies(normalised=False), [2, 2])

    dist = DiscreteDistribution({("A", "x"): 2, ("B", "y"): 2})
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist[("A", "x")] == 2
    assert dist[("B", "y")] == 2
    assert_arr(dist.frequencies(normalised=True), [2 / 4, 2 / 4])
    assert_arr(dist.frequencies(normalised=False), [2, 2])

    dist = DiscreteDistribution({("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3})
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 0
    assert_arr(dist.frequencies(normalised=True), [1 / 6, 2 / 6, 3 / 6])
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3])

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4}
    )
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 4
    assert_arr(dist.frequencies(normalised=True), [0.1, 0.2, 0.3, 0.4])
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3, 4])

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4, ("C", "y"): 5}
    )
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist[("A", "x")] == 1
    assert dist[("A", "y")] == 2
    assert dist[("B", "x")] == 3
    assert dist[("B", "y")] == 4
    assert dist[("C", "x")] == 0
    assert dist[("C", "y")] == 5
    assert_arr(
        dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
    )
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3, 4, 5])

    dist = DiscreteDistribution(
        {("A", "x"): 1, ("A", "y"): 2, ("B", "x"): 3, ("B", "y"): 4, ("C", "z"): 5}
    )
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x", "z"}


def test_three_levels_discrete_distribution():
    dist = DiscreteDistribution({("A", "y", 1): 2})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y"}
    assert dist.rvs[2].levels == {1}
    assert dist[("A", "y", 1)] == 2
    assert dist[("A", "x", 2)] == 0
    assert_arr(dist.frequencies(normalised=True), [1])
    assert_arr(dist.frequencies(normalised=False), [2])

    dist = DiscreteDistribution({("A", "x", 1): 2, ("A", "y", 1): 2})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1}
    assert dist[("A", "x", 1)] == 2
    assert dist[("A", "y", 1)] == 2
    assert_arr(dist.frequencies(normalised=True), [0.5, 0.5])
    assert_arr(dist.frequencies(normalised=False), [2, 2])

    dist = DiscreteDistribution({("A", "x", 1): 2, ("B", "y", 2): 2})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
    assert dist[("A", "x", 1)] == 2
    assert dist[("B", "y", 2)] == 2
    assert dist[("B", "y", 3)] == 0
    assert_arr(dist.frequencies(normalised=True), [0.5, 0.5])
    assert_arr(dist.frequencies(normalised=False), [2, 2])

    dist = DiscreteDistribution({("A", "x", 1): 1, ("A", "y", 2): 2, ("B", "x", 1): 3})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist["B"] == 0
    assert_arr(dist.frequencies(normalised=True), [1 / 6, 2 / 6, 3 / 6])
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3])

    dist = DiscreteDistribution(
        {("A", "x", 1): 1, ("A", "y", 2): 2, ("B", "x", 1): 3, ("B", "y", 2): 4}
    )
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist[("B", "y", 2)] == 4
    assert_arr(dist.frequencies(normalised=True), [0.1, 0.2, 0.3, 0.4])
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3, 4])

    dist = DiscreteDistribution(
        {
            ("A", "x", 1): 1,
            ("A", "y", 2): 2,
            ("B", "x", 1): 3,
            ("B", "y", 2): 4,
            ("C", "y", 3): 5,
        }
    )
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2, 3}
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 1)] == 3
    assert dist[("B", "y", 2)] == 4
    assert dist[("C", "y", 3)] == 5
    assert_arr(
        dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
    )
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3, 4, 5])

    dist = DiscreteDistribution(
        {
            ("A", "x", 1): 1,
            ("A", "y", 2): 2,
            ("B", "x", 3): 3,
            ("B", "y", 3): 4,
            ("C", "z", 4): 5,
        }
    )
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x", "z"}
    assert dist.rvs[2].levels == {1, 2, 3, 4}
    assert dist[("A", "x", 1)] == 1
    assert dist[("A", "y", 2)] == 2
    assert dist[("B", "x", 3)] == 3
    assert dist[("B", "y", 3)] == 4
    assert dist[("C", "z", 4)] == 5
    assert_arr(
        dist.frequencies(normalised=True), [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]
    )
    assert_arr(dist.frequencies(normalised=False), [1, 2, 3, 4, 5])
