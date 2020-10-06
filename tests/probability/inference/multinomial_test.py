from probability.inference import Multinomial


def test_mle_estimate_multinomial():
    sample = {"A": 15, "B": 35}
    multinomial = Multinomial.from_sample(sample, "X1")
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
