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

    # numeric key
    table = Table({1: 1, 2: 2, 3: 3})
    assert table[1] == 1
    assert table[2] == 2
    assert table[3] == 3

    table = Table({1.1: 1, 2.2: 2, 3.3: 3})
    assert table[1.1] == 1
    assert table[2.2] == 2
    assert table[3.3] == 3


def test_getitem_table():
    # One column
    table = Table({"one": 1, "two": 2, "three": 3}, names=["Y1"])
    assert table["one"] == 1
    assert table["two"] == 2
    assert table["three"] == 3
    assert table.get(Y1="one") == 1
    assert table.get(Y1="two") == 2
    assert table.get(Y1="three") == 3

    # Two columns
    table = Table(
        {("one", "Red"): 1, ("two", "Green"): 2, ("three", "Blue"): 3},
        names=["Y1", "Y2"],
    )
    # Not following the order must get None
    assert table["Red", "one"] is None
    #
    assert table["one", "Red"] == 1
    assert table["two", "Green"] == 2
    assert table["three", "Blue"] == 3
    assert table.get("one", "Red") == 1
    assert table.get("one", Y2="Red") == 1
    assert table.get("Red", Y1="one") == 1
    assert table.get(Y1="one", Y2="Red") == 1
    assert table.get("two", "Green") == 2
    assert table.get("two", Y2="Green") == 2
    assert table.get("Green", Y1="two") == 2
    assert table.get(Y1="two", Y2="Green") == 2
    assert table.get("three", "Blue") == 3
    assert table.get("three", Y2="Blue") == 3
    assert table.get("Blue", Y1="three") == 3
    assert table.get(Y1="three", Y2="Blue") == 3

    # Three columns
    table = Table(
        {
            ("one", "Red", 11): 1,
            ("two", "Green", 22): 2,
            ("three", "Blue", 33): 3,
        },
        names=["Y1", "Y2", "Y3"],
    )
    # Not following the order must get None
    assert table["Red", "one", 11] is None
    assert table[11, "Red", "one"] is None
    assert table["one", 11, "Red"] is None
    #
    assert table["one", "Red", 11] == 1
    assert table["two", "Green", 22] == 2
    assert table["three", "Blue", 33] == 3
    assert table.get("one", "Red", 11) == 1
    assert table.get("one", Y2="Red", Y3=11) == 1
    assert table.get("Red", Y1="one", Y3=11) == 1
    assert table.get(Y1="one", Y2="Red", Y3=11) == 1
    assert table.get(11, Y1="one", Y2="Red") == 1
    assert table.get(11, Y2="Red", Y1="one") == 1
    assert table.get("two", "Green", 22) == 2
    assert table.get("two", Y2="Green", Y3=22) == 2
    assert table.get("Green", Y1="two", Y3=22) == 2
    assert table.get(Y1="two", Y2="Green", Y3=22) == 2
    assert table.get("three", "Blue", 33) == 3
    assert table.get("three", Y2="Blue", Y3=33) == 3
    assert table.get("Blue", Y1="three", Y3=33) == 3
    assert table.get(Y1="three", Y2="Blue", Y3=33) == 3


def test_table_of_table():
    sample_1 = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        # ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    t1 = Table(sample_1, ["X1", "X2", "X3", "X4"])
    t2 = Table(sample_1, ["X1", "X2", "X3", "X4"])
    t3 = Table(sample_1, ["X1", "X2", "X3", "X4"])
    t4 = Table(sample_1, ["X1", "X2", "X3", "X4"])

    sample_2 = {"t1": t1, "t2": t2, "t3": t3, "t4": t4}
    tt = Table(sample_2, ["t"])
    assert tt["t2"] == t2


def test_total_table():
    table1 = Table({"a": 3, "b": 4, "c": 5}, names=["X1"])
    assert table1.total() == 12

    table1 = Table(
        {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6},
        names=["X1", "X2"],
    )
    assert table1.total() == 20

    samples = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_1 = table1.condition_on("X1", normalise=False)
    totals = con_1.total()
    assert totals[("a",)] == 36
    assert totals[("b",)] == 100


def test_normalise_table():
    table1 = Table({"a": 3, "b": 4, "c": 5}, names=["X1"])
    table1.normalise()
    assert table1["a"] == 3 / 12
    assert table1["b"] == 4 / 12

    table1 = Table(
        {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6},
        names=["X1", "X2"],
    )
    table1.normalise()
    assert table1["a", "x"] == 4 / 20
    assert table1["a", "y"] == 4 / 20
    assert table1["b", "x"] == 6 / 20
    assert table1["b", "y"] == 6 / 20

    samples = {
        ("a", "x", 1, 33): 1,
        ("a", "x", 2, 33): 2,
        ("a", "x", 1, 44): 3,
        ("a", "x", 2, 44): 4,
        ("a", "y", 1, 33): 5,
        ("a", "y", 2, 33): 6,
        ("a", "y", 1, 44): 7,
        ("a", "y", 2, 44): 8,
        ("b", "x", 1, 33): 9,
        ("b", "x", 2, 33): 10,
        ("b", "x", 1, 44): 11,
        ("b", "x", 2, 44): 12,
        ("b", "y", 1, 33): 13,
        ("b", "y", 2, 33): 14,
        ("b", "y", 1, 44): 15,
        ("b", "y", 2, 44): 16,
    }
    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_1 = table1.condition_on("X1")
    con_1.normalise()
    assert con_1["a"]["x", 1, 33] == 1 / 36
    assert con_1["a"]["y", 1, 33] == 5 / 36
    assert con_1["a"]["y", 2, 33] == 6 / 36
    assert con_1["a"]["y", 2, 44] == 8 / 36
    assert con_1["b"]["x", 1, 33] == 9 / 100
    assert con_1["b"]["y", 1, 33] == 13 / 100
    assert con_1["b"]["y", 2, 44] == 16 / 100

    con_1 = table1.condition_on("X2")
    con_1.normalise()
    assert con_1["x"]["a", 1, 33] == 1 / 52
    assert con_1["x"]["a", 2, 44] == 4 / 52
    assert con_1["y"]["b", 1, 33] == 13 / 84

    con_1 = table1.condition_on("X1", "X2")
    con_1.normalise()
    assert con_1["a", "x"][1, 33] == 1 / 10
    assert con_1["a", "x"][2, 44] == 4 / 10
    assert con_1["b", "y"][2, 33] == 14 / 58
