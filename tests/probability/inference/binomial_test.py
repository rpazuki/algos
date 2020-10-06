from probability.inference import Binomial


def test_mle_estimate_binomial():
    sample = {"A": 15, "B": 35}
    binomial = Binomial.from_sample(sample, level="A")
    assert binomial.theta == 15 / 50
    binomial = Binomial.from_sample(sample, level="B")
    assert binomial.theta == 35 / 50
