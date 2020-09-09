import pytest
import numpy as np
from probability.distributions import DiscreteDistribution
from tests.helpers import compare


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
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog"}
    assert dist["Dog"] == 2
    assert dist["Cat"] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob("Dog") == 1
    assert dist.prob(X1="Dog") == 1

    dist = DiscreteDistribution({"Dog": 2, "Cat": 3})
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog", "Cat"}
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
    assert dist.rvs.size == 1
    assert dist.rvs[0].levels == {"Dog", "Cat", "Dolphin"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y"}
    assert dist[("A", "y")] == 2
    assert dist[("A", "x")] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob("A", "y") == 1
    assert dist.prob("A", X2="y") == 1
    assert dist.prob("y", X1="A") == 1
    assert dist.prob(X1="A", X2="y") == 1

    dist = DiscreteDistribution({("A", "x"): 2, ("A", "y"): 2})
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y", "x"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x"}
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
    assert dist.rvs.size == 2
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x", "z"}
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
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y"}
    assert dist.rvs[2].levels == {1}
    assert dist[("A", "y", 1)] == 2
    assert dist[("A", "x", 2)] == 0
    assert all(compare(dist.frequencies(normalised=True), [1]))
    assert all(compare(dist.frequencies(normalised=False), [2]))
    assert dist.prob(X1="A", X2="y", X3=1) == 1
    assert dist.prob(X1="A", X2="y", X3=2) == 0

    dist = DiscreteDistribution({("A", "x", 1): 2, ("A", "y", 1): 2})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1}
    assert dist[("A", "x", 1)] == 2
    assert dist[("A", "y", 1)] == 2
    assert all(compare(dist.frequencies(normalised=True), [0.5, 0.5]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob(X1="A", X2="x", X3=1) == 0.5
    assert dist.prob(X1="A", X2="y", X3=1) == 0.5
    assert dist.prob(X1="A", X2="y", X3=2) == 0

    dist = DiscreteDistribution({("A", "x", 1): 2, ("B", "y", 2): 2})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
    assert dist[("A", "x", 1)] == 2
    assert dist[("B", "y", 2)] == 2
    assert dist[("B", "y", 3)] == 0
    assert all(compare(dist.frequencies(normalised=True), [0.5, 0.5]))
    assert all(compare(dist.frequencies(normalised=False), [2, 2]))
    assert dist.prob(X1="A", X2="x", X3=1) == 0.5
    assert dist.prob(X1="B", X2="y", X3=2) == 0.5

    dist = DiscreteDistribution({("A", "x", 1): 1, ("A", "y", 2): 2, ("B", "x", 1): 3})
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
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
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2}
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
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x"}
    assert dist.rvs[2].levels == {1, 2, 3}
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
    assert dist.rvs.size == 3
    assert dist.rvs[0].levels == {"A", "B", "C"}
    assert dist.rvs[1].levels == {"y", "x", "z"}
    assert dist.rvs[2].levels == {1, 2, 3, 4}
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
