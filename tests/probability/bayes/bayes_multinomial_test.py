from probability.empirical_distributions import FrequencyTable
from probability.bayes import Multinomial


def test_map_estimate_multinomial():
    sample = {"A": 15, "B": 35}
    empirical = FrequencyTable(sample, "X1")
    multinomial = Multinomial(empirical, 21, 21)
    assert multinomial.map()["A"] == (15 + 20) / (50 + 40)
    assert multinomial.map()["B"] == (35 + 20) / (50 + 40)
