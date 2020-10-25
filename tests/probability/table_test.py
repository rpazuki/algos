from collections import Counter
from probability.core import Table


def test_constructor_table():

    table = Table({"one": 1, "two": 2, "three": 3}, names=["Y1"])
    assert table["one"] == 1
    assert table["two"] == 2
    assert table["three"] == 3

    table = Table([("two", 2), ("one", 1), ("three", 3)])
    assert table["one"] == 1
    assert table["two"] == 2
    assert table["three"] == 3

    samples = [1, 2, 2, 3, 3, 3]
    counter = Counter(samples)
    table = Table(counter)
    assert table[1] == 1
    assert table[2] == 2
    assert table[3] == 3

    table = Table(zip(["one", "two", "three"], [1, 2, 3]))
    assert table["one"] == 1
    assert table["two"] == 2
    assert table["three"] == 3
