import pytest
from pytest import approx
from probability import Table
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

sample_3 = {
    ("a", "x", 1, 33, 20.0): 1,
    ("a", "x", 2, 33, 20.0): 2,
    ("a", "x", 1, 44, 20.0): 3,
    ("a", "x", 2, 44, 20.0): 4,
    ("a", "y", 1, 33, 20.0): 5,
    ("a", "y", 2, 33, 20.0): 6,
    ("a", "y", 1, 44, 20.0): 7,
    ("a", "y", 2, 44, 20.0): 8,
    ("b", "x", 1, 33, 20.0): 9,
    ("b", "x", 2, 33, 20.0): 10,
    ("b", "x", 1, 44, 20.0): 11,
    ("b", "x", 2, 44, 20.0): 12,
    ("b", "y", 1, 33, 20.0): 13,
    # ("b", "y", 2, 33, 20.0): 14,
    ("b", "y", 1, 44, 20.0): 15,
    ("b", "y", 2, 44, 20.0): 16,
    ("a", "x", 1, 33, 40.0): 17,
    ("a", "x", 2, 33, 40.0): 18,
    ("a", "x", 1, 44, 40.0): 19,
    ("a", "x", 2, 44, 40.0): 20,
    ("a", "y", 1, 33, 40.0): 21,
    ("a", "y", 2, 33, 40.0): 22,
    ("a", "y", 1, 44, 40.0): 23,
    ("a", "y", 2, 44, 40.0): 24,
    ("b", "x", 1, 33, 40.0): 25,
    ("b", "x", 2, 33, 40.0): 26,
    ("b", "x", 1, 44, 40.0): 27,
    ("b", "x", 2, 44, 40.0): 28,
    ("b", "y", 1, 33, 40.0): 29,
    ("b", "y", 2, 33, 40.0): 30,
    ("b", "y", 1, 44, 40.0): 31,
    ("b", "y", 2, 44, 40.0): 32,
}


def test_product_exceptions_table():
    table = Table(sample_1)
    with pytest.raises(ValueError):
        table *= "ttt"


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


def test_product_with_table_of_table_with_no_common_table():
    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    table2 = Table({"one": 1, "two": 2, "three": 3}, names=["X1"])
    con_1 = table1.condition_on("X1")
    # P(X2, X3, X4 | X1) * P(X1) -> P(X2, X3, X4, X1)
    product_1 = con_1 * table2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X2", "X3", "X4", "X1"]))
    # P(X1) * P(X2, X3, X4 | X1) -> P(X1, X2, X3, X4)
    product_1 = table2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1", "X2", "X3", "X4"]))

    table2 = Table(
        {("one", "x"): 1, ("two", "x"): 2, ("three", "y"): 3}, names=["X1", "X2"]
    )

    con_1 = table1.condition_on("X1", "X2")
    # P(X3, X4 | X1, X2) * P(X1, X2) -> P(X3, X4 , X1, X2)
    product_1 = con_1 * table2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X3", "X4", "X1", "X2"]))
    # P(X1, X2) * P(X3, X4 | X1, X2) -> P( X1, X2, X3, X4)
    product_1 = table2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1", "X2", "X3", "X4"]))

    con_2 = table2.condition_on("X1")
    # P(X3, X4 | X1, X2) * P(X2 | X1) -> P(X3, X4, X2 | X1)
    product_1 = con_1 * con_2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1"]))
    assert all(compare(product_1.columns.children_names, ["X3", "X4", "X2"]))
    # P(X2 | X1) * P(X3, X4 | X1, X2) -> P(X2, X3, X4 | X1)
    product_1 = con_2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1"]))
    assert all(compare(product_1.columns.children_names, ["X2", "X3", "X4"]))

    table3 = Table(
        {("one", "x", 1): 1, ("two", "x", 1): 2, ("three", "y", 2): 3},
        names=["X1", "X2", "X3"],
    )

    con_1 = table1.condition_on("X1", "X2", "X3")
    # P(X4 | X1, X2, X3) * P(X1, X2, X3) -> P(X4, X1, X2, X3)
    product_1 = con_1 * table3
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X4", "X1", "X2", "X3"]))
    # P(X1, X2) * P(X3, X4 | X1, X2) -> P( X1, X2, X3, X4)
    product_1 = table3 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1", "X2", "X3", "X4"]))

    con_2 = table3.condition_on("X1")
    # P(X4 | X1, X2, X3) * P(X2, X3 | X1) -> P(X4, X2, X3 | X1)
    product_1 = con_1 * con_2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1"]))
    assert all(compare(product_1.columns.children_names, ["X4", "X2", "X3"]))
    # P(X2, X3 | X1) * P(X4 | X1, X2, X3) -> P(X2, X3, X4 | X1)
    product_1 = con_2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1"]))
    assert all(compare(product_1.columns.children_names, ["X2", "X3", "X4"]))

    con_2 = table3.condition_on("X1", "X2")
    # P(X4 | X1, X2, X3) * P(X3 | X1, X2) -> P(X4, X3 | X1, X2)
    product_1 = con_1 * con_2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1", "X2"]))
    assert all(compare(product_1.columns.children_names, ["X4", "X3"]))
    # P(X3 | X1, X2) * P(X4 | X1, X2, X3) -> P(X3, X4 | X1, X2)
    product_1 = con_2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X1", "X2"]))
    assert all(compare(product_1.columns.children_names, ["X3", "X4"]))

    con_2 = table3.condition_on("X2", "X3")
    # P(X4 | X1, X2, X3) * P(X1 | X2, X3) -> P(X4, X1 | X2, X3)
    product_1 = con_1 * con_2
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X2", "X3"]))
    assert all(compare(product_1.columns.children_names, ["X4", "X1"]))
    # P(X1 | X2, X3) * P(X4 | X1, X2, X3) -> P(X1, X4 | X2, X3)
    product_1 = con_2 * con_1
    assert len(product_1) == 0
    assert all(compare(product_1.names, ["X2", "X3"]))
    assert all(compare(product_1.columns.children_names, ["X1", "X4"]))


def test_product_with_table_of_table_column_table():

    table1 = Table(sample_3, names=["X1", "X2", "X3", "X4", "X5"])

    def assert_all(table1, table2):
        for key1 in table1:
            key2_dict = table1.columns.named_key(key1)
            assert table1[key1] == approx(table2.get(**key2_dict))

    table1.normalise()

    con_1 = table1.condition_on("X1")
    table2 = table1.marginal("X2", "X3", "X4", "X5")
    # P(X2, X3, X4, X5 | X1) * P(X1) -> P(X2, X3, X4, X5, X1)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X1) * P(X2, X3, X4, X5 | X1) -> P(X1, X2, X3, X4, X5)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.condition_on("X2")
    table2 = table1.marginal("X1", "X3", "X4", "X5")
    # P(X1, X3, X4, X5 | X2) * P(X1) -> P(X1, X3, X4, X5, X2)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X1) * P(X1, X3, X4, X5 | X2) -> P(X1, X3, X4, X5, X1)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.condition_on("X5")
    table2 = table1.marginal("X1", "X2", "X3", "X4")
    # P(X1, X2, X3, X4 | X5) * P(X5) -> P(X1, X2, X3, X4, X5)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X5) * P(X1, X2, X3, X4 | X5) -> P(X5, X1, X2, X3, X4)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.condition_on("X1", "X2")
    table2 = table1.marginal("X3", "X4", "X5")
    # P(X3, X4, X5 | X1, X2) * P(X1, X2) -> P(X3, X4, X5, X1, X2)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X1, X2) * P(X3, X4, X5 | X1, X2) -> P(X1, X2, X3, X4, X5)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.condition_on("X1", "X3")
    table2 = table1.marginal("X2", "X4", "X5")
    # P(X2, X4, X5 | X1, X3) * P(X1, X3) -> P(X2, X4, X5, X1, X3)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X1, X3) * P(X2, X4, X5 | X1, X3) -> P(X1, X3, X2, X4, X5)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.condition_on("X2", "X3")
    table2 = table1.marginal("X1", "X4", "X5")
    # P(X1, X4, X5 | X2, X3) * P(X2, X3) -> P(X1, X4, X5, X2, X3)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X2, X3) * P(X1, X4, X5 | X2, X3) -> P(X2, X3, X1, X4, X5)
    product_1 = table2 * con_1
    assert_all(product_1, table1)

    con_1 = table1.marginal("X4", "X5").condition_on("X2", "X3")
    table2 = table1.condition_on("X1", "X2", "X3")
    # P(X1 | X2, X3) * P(X4, X5 | X1, X2, X3) -> P(X1, X4, X5| X2, X3)
    product_1 = con_1 * table2
    assert_all(product_1, table1)
    # P(X4, X5 | X1, X2, X3) * P(X1 | X2, X3) -> P(X4, X5, X1| X2, X3)
    product_1 = table2 * con_1
    assert_all(product_1, table1)
