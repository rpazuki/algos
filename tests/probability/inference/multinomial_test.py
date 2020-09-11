from probability.empirical_distributions import FrequencyTable
from probability.inference import Multinomial


def test_mle_estimate_multinomial():
    sample = {"A": 15, "B": 35}
    empirical = FrequencyTable(sample, "X1")
    multinomial = Multinomial(empirical)
    assert multinomial.theta_hat["A"] == 15 / 50
    assert multinomial.theta_hat["B"] == 35 / 50
