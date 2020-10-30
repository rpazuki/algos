from probability.core import Table
from tests.helpers import compare


def test_conditional_on_table():
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
    table1 = Table(samples, names=["X1", "X2", "X3", "X4"])
    con_table1 = table1.group_on("X2")
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
