from probability2.empirical_distributions import DiscreteDistribution
from tests.helpers import compare


def test_conditional_discrete_distribution():
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
    con_disc_dist = disc_dist.condition_on("X2")
    assert all(compare(con_disc_dist.conditional_rvs.names, ["X2"]))
    assert all(compare(con_disc_dist.distributions["x"].names, ["X1", "X3", "X4"]))
    assert all(compare(con_disc_dist.distributions["y"].names, ["X1", "X3", "X4"]))

    assert con_disc_dist.frequency(("a", 1, 33), "x") == 1
    assert con_disc_dist.frequency(("a", 1, 33), "y") == 5
    assert con_disc_dist.frequency(("a", 1, 44), "x") == 3
    assert con_disc_dist.frequency(("a", 1, 44), "y") == 7
    assert con_disc_dist.frequency(("b", 1, 33), "x") == 9
    assert con_disc_dist.frequency(("b", 1, 33), "y") == 13
    assert con_disc_dist.frequency(("b", 1, 44), "x") == 11
    assert con_disc_dist.frequency(("b", 1, 44), "y") == 15
    assert con_disc_dist.frequency(("b", 2, 44), "x") == 12
    assert con_disc_dist.frequency(("b", 2, 33), "y") == 14

    assert con_disc_dist.probability(("a", 1, 33), "x") == 1 / 52
    assert con_disc_dist.probability(("a", 1, 33), "y") == 5 / 84
    assert con_disc_dist.probability(("a", 1, 44), "x") == 3 / 52
    assert con_disc_dist.probability(("a", 1, 44), "y") == 7 / 84
    assert con_disc_dist.probability(("b", 1, 33), "x") == 9 / 52
    assert con_disc_dist.probability(("b", 1, 33), "y") == 13 / 84
    assert con_disc_dist.probability(("b", 1, 44), "x") == 11 / 52
    assert con_disc_dist.probability(("b", 1, 44), "y") == 15 / 84
    assert con_disc_dist.probability(("b", 2, 44), "x") == 12 / 52
    assert con_disc_dist.probability(("b", 2, 33), "y") == 14 / 84


def test_conditional_operator_discrete_distribution():
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
    con_disc_dist = disc_dist | "X2"
    assert all(compare(con_disc_dist.conditional_rvs.names, ["X2"]))
    assert all(compare(con_disc_dist.distributions["x"].names, ["X1", "X3", "X4"]))
    assert all(compare(con_disc_dist.distributions["y"].names, ["X1", "X3", "X4"]))

    assert con_disc_dist.frequency(("a", 1, 33), "x") == 1
    assert con_disc_dist.frequency(("a", 1, 33), "y") == 5

    assert con_disc_dist.probability(("a", 1, 33), "x") == 1 / 52
    assert con_disc_dist.probability(("a", 1, 33), "y") == 5 / 84

    con_disc_dist = disc_dist | ("X2", "X3")
    assert all(compare(con_disc_dist.conditional_rvs.names, ["X2", "X3"]))
    assert all(compare(con_disc_dist.distributions[("x", 1)].names, ["X1", "X4"]))
    assert all(compare(con_disc_dist.distributions[("x", 2)].names, ["X1", "X4"]))
    assert all(compare(con_disc_dist.distributions[("y", 1)].names, ["X1", "X4"]))
    assert all(compare(con_disc_dist.distributions[("y", 2)].names, ["X1", "X4"]))
    assert con_disc_dist.frequency(("a", 33), ("x", 1)) == 1
    assert con_disc_dist.probability(("a", 33), ("x", 1)) == 1 / 24
