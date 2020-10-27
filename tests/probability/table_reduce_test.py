from probability.core import Table
from tests.helpers import compare


def test_reduce_by_name_discrete_distribution():
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
    table = Table(samples)
    reduced_table = table.reduce(X2="y")
    assert reduced_table.columns.size == 3
    assert all(compare(reduced_table.names, ["X1", "X3", "X4"]))
    assert reduced_table[("a", 1, 33)] == 5
    assert reduced_table[("b", 2, 44)] == 16

    reduced_table = table.reduce(X2="y", X3=1)
    assert reduced_table.columns.size == 2
    assert all(compare(reduced_table.names, ["X1", "X4"]))
    assert reduced_table[("a", 33)] == 5
    assert reduced_table[("b", 44)] == 15

    reduced_table = table.reduce(X1="b", X3=1, X4=44)
    assert reduced_table.columns.size == 1
    assert all(compare(reduced_table.names, ["X2"]))
    assert reduced_table["x"] == 11
    assert reduced_table["y"] == 15

    table = Table(samples, names=["Y", "Z", "W", "X"])

    reduced_table = table.reduce(Z="y")
    assert reduced_table.columns.size == 3
    assert all(compare(reduced_table.names, ["Y", "W", "X"]))
    assert reduced_table[("a", 1, 33)] == 5
    assert reduced_table[("b", 2, 44)] == 16

    reduced_table = table.reduce(Z="y", W=1)
    assert reduced_table.columns.size == 2
    assert all(compare(reduced_table.names, ["Y", "X"]))
    assert reduced_table[("a", 33)] == 5
    assert reduced_table[("b", 44)] == 15

    reduced_table = table.reduce(Y="b", W=1, X=44)
    assert reduced_table.columns.size == 1
    assert all(compare(reduced_table.names, ["Z"]))
    assert reduced_table["x"] == 11
    assert reduced_table["y"] == 15
