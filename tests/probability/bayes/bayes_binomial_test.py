from probability.empirical_distributions import FrequencyTable
from probability.bayes import Binomial


def test_map_estimate_binomial():
    sample = {"A": 15, "B": 35}
    empirical = FrequencyTable(sample, "X1")
    binomial = Binomial(empirical, alpha=1, beta=1, level="A")
    assert binomial.map() == 16 / 52
    binomial = Binomial(empirical, alpha=1, beta=1, level="B")
    assert binomial.map() == 36 / 52
