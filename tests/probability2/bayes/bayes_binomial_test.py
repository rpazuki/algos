from probability2.bayes import Binomial


# def test_map_estimate_binomial():
#     sample = {"A": 15, "B": 35}
#     binomial = Binomial(sample, alpha=1, beta=1, level="A")
#     assert binomial.map == 16 / 52
#     binomial = Binomial(sample, alpha=1, beta=1, level="B")
#     assert binomial.map == 36 / 52


# def test_probability_binomial():
#     sample = {"A": 15, "B": 35}
#     binomial = Binomial(sample, alpha=1, beta=1, level="A")
#     assert binomial.probability("A") == 16 / 52
#     assert binomial.probability("B") == 36 / 52
#     assert binomial["A"] == 16 / 52
#     assert binomial["B"] == 36 / 52
#     assert binomial[0:2] == [16 / 52, 36 / 52]


# def test_probability_no_key_binomial():
#     sample = {"A": 15, "B": 35}
#     binomial = Binomial(sample, alpha=1, beta=1, level="A")
#     assert binomial.probability("AB") == 0
#     assert binomial["AB"] == 0


# def test_product_binomial():
#     sample = {"A": 15, "B": 35}
#     binomial1 = Binomial(sample, alpha=1, beta=1, level="A", name="X1")
#     binomial2 = Binomial(sample, alpha=1, beta=1, level="A", name="X2")
#     binomial3 = binomial1 * binomial2
#     print(binomial3)
#     for k in binomial3:
#         print(k, binomial3[k])

#     binomial3 = Binomial(sample, alpha=1, beta=1, level="A", name="X3")
#     binomial4 = binomial1 * binomial2 * binomial3
#     for k in binomial4:
#         print(k, binomial4[k])
