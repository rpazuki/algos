import pytest
from probability.core import Table
from tests.helpers import compare


def test_add_exception_table():
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {"a": 3, "b": 4, "c": 5}
        table1 = Table(samples, names=["X1"])
        table2 = Table(samples, names=["X2"])
        table1 += table2

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table1 = Table(samples, names=["X1", "X2"])
        table2 = Table(samples, names=["Y2", "Y2"])
        table1 += table2

    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table1 = Table(samples, names=["X1", "X2"])
        table2 = Table(samples, names=["X1", "Y2"])
        table1 += table2

    # wrong order in names
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table1 = Table(samples, names=["X1", "X2"])
        table2 = Table(samples, names=["X2", "X1"])
        table1 += table2

    sample_2 = {
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
    sample_3 = {
        ("a", 1, 33): 1,
        ("a", 2, 33): 2,
        ("a", 1, 44): 7,
        ("a", 2, 44): 8,
        ("b", 1, 33): 9,
        ("b", 2, 33): 10,
        ("b", 2, 44): 16,
    }
    # conditional
    with pytest.raises(ValueError):
        table1 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        table2 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1")
        con_2 = table2.condition_on("X2")
        con_1 += con_2

    with pytest.raises(ValueError):
        table1 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        table2 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        con_1 = table1.condition_on("X1", "X2")
        con_2 = table2.condition_on("X2")
        con_1 += con_2

    with pytest.raises(ValueError):
        table1 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        table2 = Table(sample_3, names=["X1", "X2", "X3"])
        con_1 = table1.condition_on("X1")
        con_2 = table2.condition_on("X1")
        con_1 += con_2

    with pytest.raises(ValueError):
        table1 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
        table2 = Table(sample_3, names=["X1", "X2", "X3"])
        con_1 = table1.condition_on("X1", "X2")
        con_2 = table2.condition_on("X1", "X2")
        con_1 += con_2


def test_add_table():
    samples = {"a": 3, "b": 4, "c": 5}
    table1 = Table(samples, names=["X1"])
    table2 = Table(samples, names=["X1"])
    table3 = table1 + table2
    assert table3["a"] == 2 * 3
    assert table3["b"] == 2 * 4
    assert table3["c"] == 2 * 5

    samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
    table1 = Table(samples, names=["X1", "X2"])
    table2 = Table(samples, names=["X1", "X2"])
    table3 = table1 + table2
    assert table3["a", "x"] == 2 * 4
    assert table3["a", "y"] == 2 * 4
    assert table3["b", "x"] == 2 * 6
    assert table3["b", "y"] == 2 * 6

    sample_2 = {
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

    table1 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
    table2 = Table(sample_2, names=["X1", "X2", "X3", "X4"])
    table3 = table1 + table2
    assert table3["a", "x", 1, 33] == 2 * 1
    assert table3["a", "y", 2, 33] == 2 * 6
    assert table3["b", "x", 2, 44] == 2 * 12
    assert table3[("b", "y", 2, 44)] == 2 * 16

    con_1 = table1.condition_on("X1")
    con_2 = table2.condition_on("X1")
    table3 = con_1 + con_2
    assert table3["a"]["x", 1, 33] == 2 * 1
    assert table3["a"]["x", 1, 44] == 2 * 3
    assert table3["b"]["x", 1, 33] == 2 * 9
    assert table3["b"]["x", 1, 44] == 2 * 11

    con_1 = table1.condition_on("X1", "X3")
    con_2 = table2.condition_on("X1", "X3")
    table3 = con_1 + con_2
    assert table3["a", 1]["x", 33] == 2 * 1
    assert table3["a", 2]["y", 44] == 2 * 8
    assert table3["b", 1]["x", 44] == 2 * 11
    assert table3["b", 2]["y", 33] is None
    assert table3["b", 2]["y", 44] == 2 * 16
