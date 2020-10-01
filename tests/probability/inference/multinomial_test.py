from probability.inference import Multinomial


def test_mle_estimate_multinomial():
    sample = {"A": 15, "B": 35}
    multinomial = Multinomial(sample, "X1")
    assert multinomial.theta_hat["A"] == 15 / 50
    assert multinomial.theta_hat["B"] == 35 / 50
