from probability import Table
from probability.inference.bayes import Multinomial


def test_multinomial_bays():
    data = Table({"A": 0, "B": 1}, names=["X1"])
    b1 = Multinomial(data)
    print(b1)
    print(b1["A"])
    print(b1["B"])

    data = Table(
        {("A", 1): 1, ("B", 1): 2, ("A", 2): 3, ("B", 2): 4}, names=["X1", "X2"]
    )
    b1 = Multinomial(data)
    print(b1)
    print(b1[("A", 1)])
    print(b1["B", 1])
    print(b1[("A", 2)])
    print(b1["B", 2])
    print(b1["B", 3])

    print(b1.marginal("X2"))
