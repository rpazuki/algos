import numpy as np
from pytest import approx
from probability2.empirical_distributions import FrequencyTable
from probability2.empirical_distributions import DiscreteDistribution
from information_theory.measures import entropy


def test_entropy():
    # Binary distribution from samples
    # This is 50-50 samples with entropy = log2(2)
    samples = [1, 1, 2, 2, 1, 1, 2, 2]
    ft = FrequencyTable(samples)
    assert entropy(ft) == np.log2(2)

    dd = DiscreteDistribution(samples)
    assert entropy(dd) == np.log2(2)

    # This is 60-40 samples with entropy = 0.970950
    samples = ["Dog", "Dog", "Dog", "Dog", "Dog", "Dog", "Cat", "Cat", "Cat", "Cat"]
    ft = FrequencyTable(samples)
    assert entropy(ft) == approx(0.970950)

    dd = DiscreteDistribution(samples)
    assert entropy(dd) == approx(0.970950)

    # Deterministic case
    samples = {"Dog": 10, "Cat": 0}
    ft = FrequencyTable(samples)
    assert entropy(ft) == 0

    dd = DiscreteDistribution(samples)
    assert entropy(dd) == 0

    # Multiple levels
    samples = {(1, 2): 150, (1, 3): 150, (2, 2): 300, (2, 3): 400}
    dd = DiscreteDistribution(samples)
    assert entropy(dd) == approx(1.8709505945)
