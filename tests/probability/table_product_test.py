import pytest
from probability.core import Table
from tests.helpers import compare

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

sample_2 = {
    (1, "high", "under", "x"): 1,
    (1, "high", "normal", "x"): 2,
    (1, "high", "over", "x"): 3,
    (1, "high", "obese", "x"): 4,
    (1, "low", "under", "x"): 5,
    (1, "low", "normal", "x"): 6,
    (1, "low", "over", "x"): 7,
    (1, "low", "obese", "x"): 8,
    (2, "high", "under", "x"): 9,
    (2, "high", "normal", "x"): 10,
    (2, "high", "over", "x"): 11,
    (2, "high", "obese", "x"): 12,
    (2, "low", "under", "x"): 13,
    (2, "low", "normal", "x"): 14,
    (2, "low", "over", "x"): 15,
    (2, "low", "obese", "x"): 16,
    (1, "high", "under", "y"): 17,
    (1, "high", "normal", "y"): 18,
    (1, "high", "over", "y"): 19,
    (1, "high", "obese", "y"): 20,
    (1, "low", "under", "y"): 21,
    (1, "low", "normal", "y"): 22,
    (1, "low", "over", "y"): 23,
    (1, "low", "obese", "y"): 24,
    (2, "high", "under", "y"): 25,
    (2, "high", "normal", "y"): 26,
    # (2, "high", "over", "y"): 27,
    # (2, "high", "obese", "y"): 28,
    # (2, "low", "under", "y"): 29,
    # (2, "low", "normal", "y"): 30,
    # (2, "low", "over", "y"): 31,
    # (2, "low", "obese", "y"): 32,
}


def test_product_exceptions_table():
    table = Table(sample_1)
    with pytest.raises(ValueError):
        table *= "ttt"


def test_product_with_a_number_table():
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = 2 * table1
    assert table2["a", "x", 1, 33] == 1 * 2
    assert table2["b", "x", 1, 33] == 9 * 2
    assert table2["b", "x", 2, 44] == 12 * 2

    table3 = table1 * 1.2
    assert table3["a", "x", 1, 33] == 1 * 1.2
    assert table3["b", "x", 1, 33] == 9 * 1.2
    assert table3["b", "x", 2, 44] == 12 * 1.2


def test_product_with_one_common_var_table():

    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = Table(sample_2, names=["X3", "X5", "X6", "X7"])

    table3 = table1 * table2
    assert all(compare(table3.names, ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]))

    # check probabilites
    assert table3["a", "x", 1, 33, "high", "normal", "x"] == 2
    # check the case that the right does not have the common
    assert table3[("b", "y", 2, 44, "high", "over", "y")] is None
    # check the case that the left does not have the common
    assert table3[("b", "y", 2, 33, "high", "normal", "y")] is None


def test_product_with_two_common_vars_table():
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = Table(sample_2, names=["X3", "X5", "X6", "X2"])

    table3 = table1 * table2
    assert all(compare(table3.names, ["X1", "X2", "X3", "X4", "X5", "X6"]))

    # check probabilites
    assert table3[("a", "y", 2, 33, "high", "under")] == 150
    # check the case that the right does not have the common
    assert table3[("a", "y", 2, 33, "low", "under")] is None
    # check the case that the left does not have the common
    assert table3[("b", "y", 2, 33, "high", "under")] is None


def test_product_with_no_common_vars_table():

    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = Table(sample_2, names=["Y1", "Y2", "Y3", "Y4"])

    table3 = table1 * table2
    assert all(compare(table3.names, ["X1", "X2", "X3", "X4", "Y1", "Y2", "Y3", "Y4"]))

    # check probabilites
    assert table3[("a", "x", 1, 33, 2, "high", "normal", "x")] == 10

    assert table3[("b", "x", 1, 44, 1, "low", "over", "y")] == 253


def test_product_with_single_column_table():
    single_table = Table({"A": 3, "B": 4, "C": 7}, names=["Y1"])
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])

    # without common names
    table3 = table1 * single_table
    assert all(compare(table3.names, ["X1", "X2", "X3", "X4", "Y1"]))

    # check probabilites
    assert table3[("a", "y", 2, 33, "B")] == 24

    table3 = single_table * table1
    assert all(compare(table3.names, ["Y1", "X1", "X2", "X3", "X4"]))

    # check probabilites
    assert table3[("B", "a", "y", 2, 33)] == 24

    # with common names
    single_table = Table({"x": 3, "y": 4}, names=["X2"])
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])

    table3 = table1 * single_table
    assert all(compare(table3.names, ["X1", "X2", "X3", "X4"]))

    # check probabilites
    assert table3["a", "x", 2, 44] == 12
    assert table3["a", "y", 2, 44] == 32

    table3 = single_table * table1
    assert all(compare(table3.names, ["X2", "X1", "X3", "X4"]))


def test_product_with_table_of_table_column_table():
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = Table(sample_2, names=["X3", "X5", "X6", "X7"])
    con_1 = table1.group_on("X1")
    con_2 = 2 * con_1
    for k in con_1:
        assert con_2[k] == con_1[k] * 2

    con_2 = con_1 * 2
    for k in con_1:
        assert con_2[k] == con_1[k] * 2

    con_2 = table2 * con_1
    print(con_2.names)
