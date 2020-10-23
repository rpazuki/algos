import pytest
from probability.inference import Multinomial
from probability.inference import MultinomialMLEInferrer as MLE


def test_exceptions_multinomial_mle_inferrer():

    with pytest.raises(ValueError):
        MLE(empirical_distribution=None)

    with pytest.raises(ValueError):
        MLE(empirical_distribution="None")


def test_mle_estimate_multinomial():
    sample = {"A": 15, "B": 35}
    mle = MLE.from_sample(sample)
    multinomial = Multinomial(mle)
    assert multinomial["A"] == 15 / 50
    assert multinomial.prob("A") == 15 / 50
    assert multinomial.probability("A") == 15 / 50
    assert multinomial["B"] == 35 / 50
    assert multinomial.prob("B") == 35 / 50
    assert multinomial.probability("B") == 35 / 50
    assert "A" in multinomial
    assert "B" in multinomial
    assert "C" not in multinomial
    for level, prob in multinomial.items():
        if level == "A":
            assert prob == 15 / 50
        else:
            assert prob == 35 / 50


def test_mle_estimate_multilevels_multinomial():
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
    mle = MLE.from_multilevels_sample(samples)
    multinomial = Multinomial(mle)
    assert multinomial[("a", "x", 1, 33, 1.5)] == 1 / 136
    assert multinomial.prob("a", "x", 1, 33, 1.5) == 1 / 136
    assert multinomial.probability(("a", "x", 1, 33, 1.5)) == 1 / 136
    assert multinomial[("b", "y", 1, 44, 3.5)] == 15 / 136
    assert multinomial.prob("b", "y", 1, 44, 3.5) == 15 / 136
    assert multinomial.probability(("b", "y", 1, 44, 3.5)) == 15 / 136
    assert ("a", "x", 1, 33, 1.5) in multinomial
    assert ("b", "x", 2, 44, 2.5) in multinomial
    assert ("b", "x", 2, 445, 2.5) not in multinomial
