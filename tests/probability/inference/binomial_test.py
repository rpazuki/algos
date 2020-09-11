from probability.empirical_distributions import FrequencyTable
from probability.inference import Binomial


def test_mle_estimate_binomial():
    sample = {"A": 15, "B": 35}
    empirical = FrequencyTable(sample, "X1")
    binomial = Binomial(empirical, level="A")
    assert binomial.theta_hat == 15 / 50
    binomial = Binomial(empirical, level="B")
    assert binomial.theta_hat == 35 / 50
