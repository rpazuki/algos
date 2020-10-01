from probability.inference import Binomial


def test_mle_estimate_binomial():
    sample = {"A": 15, "B": 35}
    binomial = Binomial(sample, level="A")
    assert binomial.theta_hat == 15 / 50
    binomial = Binomial(sample, level="B")
    assert binomial.theta_hat == 35 / 50
