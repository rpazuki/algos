import pytest
from probability.core import Table
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
    assert child_table1["a", 1, 33] == 1
    assert child_table2["a", 1, 33] == 5
    assert child_table1["a", 1, 44] == 3
    assert child_table2["a", 1, 44] == 7
    assert child_table1["b", 1, 33] == 9
    assert child_table2["b", 1, 33] == 13
    assert child_table1["b", 1, 44] == 11
    assert child_table2["b", 1, 44] == 15
    assert child_table1["b", 2, 44] == 12
    assert child_table2["b", 2, 33] == 14

    # combined indexing
    assert con_table1["x"]["a", 1, 33] == 1
    assert con_table1["y"]["a", 1, 33] == 5


def test_conditional_on_conditional_table():
    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_1 = table1.condition_on("X3")
    con_2 = con_1.condition_on("X1")
    # Note: since we first condition on X3
    # and then on X1, the order of keys
    # is as X1, X3. Otherwise, it could be
    # the inverse
    assert con_2["a", 1]["x", 33] == 1
    assert con_2["a", 2]["x", 33] == 2
    assert con_2["a", 1]["x", 44] == 3
    assert con_2["a", 2]["x", 44] == 4
    assert con_2["b", 1]["x", 33] == 9
    assert con_2["b", 2]["x", 33] == 10
    assert con_2["b", 1]["y", 33] == 13
    assert con_2["b", 2]["y", 44] == 16
    # Try the inverse conditioning
    con_1 = table1.condition_on("X1")
    con_2 = con_1.condition_on("X3")
    assert con_2[1, "a"]["x", 33] == 1
    assert con_2[2, "a"]["x", 44] == 4
    assert con_2[1, "b"]["x", 33] == 9
    assert con_2[2, "b"]["x", 33] == 10
    # we can use 'get' to make it order-agnostic
    assert con_2.get(X1="a", X3=1)["x", 33] == 1
    #
    #
    con_1 = table1.condition_on("X3", "X4")
    con_2 = con_1.condition_on("X1")
    assert con_2["a", 1, 33]["x"] == 1
    assert con_2["a", 2, 33]["x"] == 2
    assert con_2["a", 1, 44]["x"] == 3
    assert con_2["a", 2, 44]["x"] == 4
    assert con_2["b", 1, 44]["x"] == 11
    assert con_2["b", 2, 44]["x"] == 12
    assert con_2["b", 1, 44]["y"] == 15
    assert con_2["b", 2, 44]["y"] == 16

    con_1 = table1.condition_on("X4")
    con_2 = con_1.condition_on("X1", "X3")
    assert con_2["a", 1, 33]["x"] == 1
    assert con_2["a", 2, 33]["x"] == 2
    assert con_2["a", 1, 44]["x"] == 3
    assert con_2["a", 2, 44]["x"] == 4
    assert con_2["b", 1, 44]["x"] == 11
    assert con_2["b", 2, 44]["x"] == 12
    assert con_2["b", 1, 44]["y"] == 15
    assert con_2["b", 2, 44]["y"] == 16
    # change the order
    con_1 = table1.condition_on("X3")
    con_2 = con_1.condition_on("X1", "X4")
    assert con_2["a", 33, 1]["x"] == 1
    assert con_2["a", 33, 2]["x"] == 2
    assert con_2["a", 44, 1]["x"] == 3
    assert con_2["a", 44, 2]["x"] == 4
    assert con_2["b", 44, 1]["x"] == 11
    assert con_2["b", 44, 2]["x"] == 12
    assert con_2["b", 44, 1]["y"] == 15
    assert con_2["b", 44, 2]["y"] == 16
