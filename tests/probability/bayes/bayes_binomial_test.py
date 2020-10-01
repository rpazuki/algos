from probability.bayes import Binomial


def test_map_estimate_binomial():
    sample = {"A": 15, "B": 35}
    binomial = Binomial(sample, alpha=1, beta=1, level="A")
    assert binomial.map() == 16 / 52
    binomial = Binomial(sample, alpha=1, beta=1, level="B")
    assert binomial.map() == 36 / 52
