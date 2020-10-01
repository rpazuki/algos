import pytest
from pytest import approx
from probability import Table
from tests.helpers import compare

# Four levels dist.
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


def test_conditional_on_exception_table():
    #
    with pytest.raises(ValueError):
        table1 = Table({"a": 3, "b": 4, "c": 5}, names=["X1"])
        table1.condition_on("X1")

    with pytest.raises(ValueError):
        table1 = Table(
            {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6},
            names=["X1", "X2"],
        )
        table1.condition_on("X1", "X2")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        table1.condition_on("X1", "X2", "X3", "X4")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1")
        con_1.condition_on("X1")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1", "X2")
        con_1.condition_on("X1", "X2")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1", "X2")
        con_1.condition_on("X2", "X1")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1", "X2", "X3")
        con_1.condition_on("X1", "X2", "X3")

    with pytest.raises(ValueError):
        table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X3", "X2", "X2")
        con_1.condition_on("X1", "X3", "X1")


def test_conditional_on_table():

    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_table1 = table1.condition_on("X2")
    assert all(compare(con_table1.names, ["X2"]))
    for x2 in con_table1.keys():
        child_table = con_table1[x2]
        assert all(compare(child_table.names, ["X1", "X3", "X4"]))

    child_table1 = con_table1["x"]
    child_table2 = con_table1["y"]
    assert child_table1["a", 1, 33] == 1 / 52
    assert child_table2["a", 1, 33] == 5 / 84
    assert child_table1["a", 1, 44] == 3 / 52
    assert child_table2["a", 1, 44] == 7 / 84
    assert child_table1["b", 1, 33] == 9 / 52
    assert child_table2["b", 1, 33] == 13 / 84
    assert child_table1["b", 1, 44] == 11 / 52
    assert child_table2["b", 1, 44] == 15 / 84
    assert child_table1["b", 2, 44] == 12 / 52
    assert child_table2["b", 2, 33] == 14 / 84

    # combined indexing
    assert con_table1["x"]["a", 1, 33] == 1 / 52
    assert con_table1["y"]["a", 1, 33] == 5 / 84


def test_conditional_on_conditional_table():
    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_1 = table1.condition_on("X3")
    con_2 = con_1.condition_on("X1")
    # Note: since we first condition on X3
    # and then on X1, the order of keys
    # is as X1, X3. Otherwise, it could be
    # the inverse
    assert con_2["a", 1]["x", 33] == approx(1 / 16)
    assert con_2["a", 2]["x", 33] == approx(2 / 20)
    assert con_2["a", 1]["x", 44] == approx(3 / 16)
    assert con_2["a", 2]["x", 44] == approx(4 / 20)
    assert con_2["b", 1]["x", 33] == approx(9 / 48)
    assert con_2["b", 2]["x", 33] == approx(10 / 52)
    assert con_2["b", 1]["y", 33] == approx(13 / 48)
    assert con_2["b", 2]["y", 44] == approx(16 / 52)
    # Try the inverse conditioning
    con_1 = table1.condition_on("X1")
    con_2 = con_1.condition_on("X3")
    assert con_2[1, "a"]["x", 33] == approx(1 / 16)
    assert con_2[2, "a"]["x", 44] == approx(4 / 20)
    assert con_2[1, "b"]["x", 33] == approx(9 / 48)
    assert con_2[2, "b"]["x", 33] == approx(10 / 52)
    # we can use 'get' to make it order-agnostic
    assert con_2.get(X1="a", X3=1)["x", 33] == approx(1 / 16)
    #
    #
    con_1 = table1.condition_on("X3", "X4")
    con_2 = con_1.condition_on("X1")
    assert con_2["a", 1, 33]["x"] == approx(1 / 6)
    assert con_2["a", 2, 33]["x"] == approx(2 / 8)
    assert con_2["a", 1, 44]["x"] == approx(3 / 10)
    assert con_2["a", 2, 44]["x"] == approx(4 / 12)
    assert con_2["b", 1, 44]["x"] == approx(11 / 26)
    assert con_2["b", 2, 44]["x"] == approx(12 / 28)
    assert con_2["b", 1, 44]["y"] == approx(15 / 26)
    assert con_2["b", 2, 44]["y"] == approx(16 / 28)

    con_1 = table1.condition_on("X4")
    con_2 = con_1.condition_on("X1", "X3")
    assert con_2["a", 1, 33]["x"] == approx(1 / 6)
    assert con_2["a", 2, 33]["x"] == approx(2 / 8)
    assert con_2["a", 1, 44]["x"] == approx(3 / 10)
    assert con_2["a", 2, 44]["x"] == approx(4 / 12)
    assert con_2["b", 1, 44]["x"] == approx(11 / 26)
    assert con_2["b", 2, 44]["x"] == approx(12 / 28)
    assert con_2["b", 1, 44]["y"] == approx(15 / 26)
    assert con_2["b", 2, 44]["y"] == approx(16 / 28)
    # change the order
    con_1 = table1.condition_on("X3")
    con_2 = con_1.condition_on("X1", "X4")
    assert con_2["a", 33, 1]["x"] == approx(1 / 6)
    assert con_2["a", 33, 2]["x"] == approx(2 / 8)
    assert con_2["a", 44, 1]["x"] == approx(3 / 10)
    assert con_2["a", 44, 2]["x"] == approx(4 / 12)
    assert con_2["b", 44, 1]["x"] == approx(11 / 26)
    assert con_2["b", 44, 2]["x"] == approx(12 / 28)
    assert con_2["b", 44, 1]["y"] == approx(15 / 26)
    assert con_2["b", 44, 2]["y"] == approx(16 / 28)
