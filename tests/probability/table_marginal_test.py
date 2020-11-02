import pytest
from pytest import approx
from probability import Table
from tests.helpers import compare


def test_marginals_names_exception_table():
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {"a": 3, "b": 4, "c": 5}
        table = Table(samples)
        table.marginal("X1")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples)
        table.marginal("X0")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples)
        table.marginal("X3")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples)
        table2 = table.marginal("X1")
        table2.marginal("X1")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples, names=["Y", "Z"])
        table.marginal("X1")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples, names=["Y", "Z"])
        table.marginal("X1")
    # Wrong column name
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples, names=["Y", "Z"])
        table2 = table.marginal("Y")
        table2.marginal("Y")

    # Marginalize over all columns
    with pytest.raises(ValueError):
        samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
        table = Table(samples, names=["Y", "Z"])
        table2 = table.marginal("Y", "Z")


def test_marginals_names_table():
    samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
    table = Table(samples)

    table2 = table.marginal("X1")
    assert all(compare(table2.names, ["X2"]))

    table2 = table.marginal("X2")
    assert all(compare(table2.names, ["X1"]))
    #
    table = Table(samples, names=["Y", "Z"])

    table2 = table.marginal("Y")
    assert all(compare(table2.names, ["Z"]))

    table2 = table.marginal("Z")
    assert all(compare(table2.names, ["Y"]))

    # Three levels dist.
    samples = {
        ("a", "x", 1): 4,
        ("a", "x", 2): 4,
        ("a", "y", 1): 6,
        ("a", "y", 2): 6,
        ("b", "x", 1): 8,
        ("b", "x", 2): 8,
        ("b", "y", 1): 10,
        ("b", "y", 2): 10,
    }

    table = Table(samples)

    table2 = table.marginal("X1")
    assert all(compare(table2.names, ["X2", "X3"]))

    table2 = table.marginal("X2")
    assert all(compare(table2.names, ["X1", "X3"]))

    table2 = table.marginal("X3")
    assert all(compare(table2.names, ["X1", "X2"]))

    table2 = table.marginal("X1", "X3")
    assert all(compare(table2.names, ["X2"]))

    table2 = table.marginal("X2", "X3")
    assert all(compare(table2.names, ["X1"]))

    #
    table = Table(samples, names=["Y", "Z", "W"])

    table2 = table.marginal("Y")
    assert all(compare(table2.names, ["Z", "W"]))

    table2 = table.marginal("Z")
    assert all(compare(table2.names, ["Y", "W"]))

    table2 = table.marginal("W")
    assert all(compare(table2.names, ["Y", "Z"]))

    table2 = table.marginal("Y", "W")
    assert all(compare(table2.names, ["Z"]))

    table2 = table.marginal("Z", "W")
    assert all(compare(table2.names, ["Y"]))


def test_marginals_table():
    # Single RV dist.
    with pytest.raises(ValueError):
        table = Table({"A": 2, "B": 3, "C": 4})
        table.marginal("X1")

    # Two levels dist.
    samples = {(1, 1): 4, (1, 2): 4, (2, 1): 6, (2, 2): 6}
    table = Table(samples)
    table2 = table.marginal("X1")
    assert all(compare(table2.keys(), [(1,), (2,)]))
    assert table2[1] == 10 / 20
    assert table2[2] == 10 / 20

    table2 = table.marginal("X2")
    assert all(compare(table2.keys(), [(1,), (2,)]))
    assert table2[1] == 8 / 20
    assert table2[2] == 12 / 20

    samples = {("a", "x"): 4, ("a", "y"): 4, ("b", "x"): 6, ("b", "y"): 6}
    table = Table(samples)
    table2 = table.marginal("X1")
    assert all(compare(table2.keys(), [("x",), ("y",)]))
    assert table2["x"] == 10 / 20
    assert table2["y"] == 10 / 20

    table2 = table.marginal("X1")
    assert all(compare(table2.keys(), [("x",), ("y",)]))
    assert table2["x"] == 10 / 20
    assert table2["y"] == 10 / 20

    table2 = table.marginal("X2")
    assert all(compare(table2.keys(), [("a",), ("b",)]))
    assert table2["a"] == 8 / 20
    assert table2["b"] == 12 / 20

    # Three levels dist.
    samples = {
        ("a", "x", 1): 4,
        ("a", "x", 2): 4,
        ("a", "y", 1): 6,
        ("a", "y", 2): 6,
        ("b", "x", 1): 8,
        ("b", "x", 2): 8,
        ("b", "y", 1): 10,
        ("b", "y", 2): 10,
    }
    table = Table(samples)
    table2 = table.marginal("X1")
    assert all(compare(table2.keys(), [("x", 1), ("x", 2), ("y", 1), ("y", 2)]))
    assert table2[("x", 1)] == 12 / 56
    assert table2[("x", 2)] == 12 / 56
    assert table2[("y", 1)] == 16 / 56
    assert table2[("y", 2)] == 16 / 56

    table2 = table.marginal("X2")
    assert all(compare(table2.keys(), [("a", 1), ("a", 2), ("b", 1), ("b", 2)]))
    assert table2[("a", 1)] == 10 / 56
    assert table2[("a", 2)] == 10 / 56
    assert table2[("b", 1)] == 18 / 56
    assert table2[("b", 2)] == 18 / 56

    table2 = table.marginal("X3")
    assert all(compare(table2.keys(), [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]))
    assert table2[("a", "x")] == 8 / 56
    assert table2[("a", "y")] == 12 / 56
    assert table2[("b", "x")] == 16 / 56
    assert table2[("b", "y")] == 20 / 56

    table2 = table.marginal("X1", "X2")
    assert all(compare(table2.keys(), [(1,), (2,)]))
    assert table2[1] == 28 / 56
    assert table2[2] == 28 / 56

    table2 = table.marginal("X1", "X3")
    assert all(compare(table2.keys(), [("x",), ("y",)]))
    assert table2["x"] == 24 / 56
    assert table2["y"] == 32 / 56

    table2 = table.marginal("X2", "X3")
    assert all(compare(table2.keys(), [("a",), ("b",)]))
    assert table2["a"] == 20 / 56
    assert table2["b"] == 36 / 56

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
    table = Table(samples)
    table2 = table.marginal("X3")
    assert all(
        compare(
            table2.keys(),
            [
                ("a", "x", 33),
                ("a", "x", 44),
                ("a", "y", 33),
                ("a", "y", 44),
                ("b", "x", 33),
                ("b", "x", 44),
                ("b", "y", 33),
                ("b", "y", 44),
            ],
        )
    )
    assert table2[("a", "x", 33)] == 3 / 136
    assert table2[("a", "x", 44)] == 7 / 136
    assert table2[("a", "y", 33)] == 11 / 136
    assert table2[("a", "y", 44)] == 15 / 136
    assert table2[("b", "x", 33)] == 19 / 136
    assert table2[("b", "x", 44)] == 23 / 136
    assert table2[("b", "y", 33)] == 27 / 136
    assert table2[("b", "y", 44)] == 31 / 136

    table2 = table.marginal("X4")
    assert all(
        compare(
            table2.keys(),
            [
                ("a", "x", 1),
                ("a", "x", 2),
                ("a", "y", 1),
                ("a", "y", 2),
                ("b", "x", 1),
                ("b", "x", 2),
                ("b", "y", 1),
                ("b", "y", 2),
            ],
        )
    )
    assert table2[("a", "x", 1)] == 4 / 136
    assert table2[("a", "x", 2)] == 6 / 136
    assert table2[("a", "y", 1)] == 12 / 136
    assert table2[("a", "y", 2)] == 14 / 136
    assert table2[("b", "x", 1)] == 20 / 136
    assert table2[("b", "x", 2)] == 22 / 136
    assert table2[("b", "y", 1)] == 28 / 136
    assert table2[("b", "y", 2)] == 30 / 136

    table2 = table.marginal("X1", "X4")
    assert all(compare(table2.keys(), [("x", 1), ("x", 2), ("y", 1), ("y", 2)]))
    assert table2[("x", 1)] == 24 / 136
    assert table2[("x", 2)] == 28 / 136
    assert table2[("y", 1)] == 40 / 136
    assert table2[("y", 2)] == 44 / 136

    table2 = table.marginal("X1", "X2", "X4")
    assert all(compare(table2.keys(), [(1,), (2,)]))
    assert table2[1] == 64 / 136
    assert table2[2] == 72 / 136

    # marginalize two times
    table2 = table.marginal("X1", "X4")
    table3 = table2.marginal("X2")
    assert all(compare(table3.keys(), [(1,), (2,)]))
    assert table3[1] == 64 / 136
    assert table3[2] == 72 / 136

    # marginalize three times
    table2 = table.marginal("X4")
    table3 = table2.marginal("X3")
    table4 = table3.marginal("X2")
    assert all(compare(table4.keys(), [("a",), ("b",)]))
    assert table4["a"] == 36 / 136
    assert table4["b"] == 100 / 136


def test_marginal_by_name_table():
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
    table = Table(samples, names=["Age", "Sex", "Edu", "Etn"])
    table2 = table.marginal("Edu")
    assert all(
        compare(
            table2.keys(),
            [
                ("a", "x", 33),
                ("a", "x", 44),
                ("a", "y", 33),
                ("a", "y", 44),
                ("b", "x", 33),
                ("b", "x", 44),
                ("b", "y", 33),
                ("b", "y", 44),
            ],
        )
    )
    assert table2[("a", "x", 33)] == 3 / 136
    assert table2[("a", "x", 44)] == 7 / 136
    assert table2[("a", "y", 33)] == 11 / 136
    assert table2[("a", "y", 44)] == 15 / 136
    assert table2[("b", "x", 33)] == 19 / 136
    assert table2[("b", "x", 44)] == 23 / 136
    assert table2[("b", "y", 33)] == 27 / 136
    assert table2[("b", "y", 44)] == 31 / 136

    table2 = table.marginal("Edu", normalise=False)
    assert table2[("a", "x", 33)] == 3
    assert table2[("a", "x", 44)] == 7
    assert table2[("a", "y", 33)] == 11
    assert table2[("a", "y", 44)] == 15
    assert table2[("b", "x", 33)] == 19
    assert table2[("b", "x", 44)] == 23
    assert table2[("b", "y", 33)] == 27
    assert table2[("b", "y", 44)] == 31

    table2 = table.marginal("Etn")
    assert all(
        compare(
            table2.keys(),
            [
                ("a", "x", 1),
                ("a", "x", 2),
                ("a", "y", 1),
                ("a", "y", 2),
                ("b", "x", 1),
                ("b", "x", 2),
                ("b", "y", 1),
                ("b", "y", 2),
            ],
        )
    )
    assert table2[("a", "x", 1)] == 4 / 136
    assert table2[("a", "x", 2)] == 6 / 136
    assert table2[("a", "y", 1)] == 12 / 136
    assert table2[("a", "y", 2)] == 14 / 136
    assert table2[("b", "x", 1)] == 20 / 136
    assert table2[("b", "x", 2)] == 22 / 136
    assert table2[("b", "y", 1)] == 28 / 136
    assert table2[("b", "y", 2)] == 30 / 136

    table2 = table.marginal("Age", "Etn")
    assert all(compare(table2.keys(), [("x", 1), ("x", 2), ("y", 1), ("y", 2)]))
    assert table2[("x", 1)] == 24 / 136
    assert table2[("x", 2)] == 28 / 136
    assert table2[("y", 1)] == 40 / 136
    assert table2[("y", 2)] == 44 / 136

    table2 = table.marginal("Age", "Sex", "Etn")
    assert all(compare(table2.keys(), [(1,), (2,)]))
    assert table2[1] == 64 / 136
    assert table2[2] == 72 / 136

    # marginalize two times
    table2 = table.marginal("Age", "Etn")
    table3 = table2.marginal("Sex")
    assert all(compare(table3.keys(), [(1,), (2,)]))
    assert table3[1] == 64 / 136
    assert table3[2] == 72 / 136

    # marginalize three times
    table2 = table.marginal("Etn")
    table3 = table2.marginal("Edu")
    table4 = table3.marginal("Sex")
    assert all(compare(table4.keys(), [("a",), ("b",)]))
    assert table4["a"] == 36 / 136
    assert table4["b"] == 100 / 136


def test_marginal_if_table_of_table():
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

    table1 = Table(sample_1, names=["X1", "X2", "X3", "X4"])
    con_1 = table1.condition_on("X1")
    with pytest.raises(ValueError):
        con_1.marginal("X1")

    marginal_1 = con_1.marginal("X2")
    assert all(compare(marginal_1.names, ["X1"]))
    assert all(compare(marginal_1.children_names, ["X3", "X4"]))
    assert marginal_1["a"][1, 33] == approx((1 + 5) / 36)
    assert marginal_1["a"][1, 44] == approx((3 + 7) / 36)
    assert marginal_1["a"][2, 33] == approx((2 + 6) / 36)
    assert marginal_1["a"][2, 44] == approx((4 + 8) / 36)
    assert marginal_1["b"][1, 33] == approx((9 + 13) / 86)
    assert marginal_1["b"][1, 44] == approx((11 + 15) / 86)

    con_1 = table1.condition_on("X1", normalise=False)
    marginal_1 = con_1.marginal("X2", normalise=False)
    assert marginal_1["a"][1, 33] == (1 + 5)
    assert marginal_1["a"][1, 44] == (3 + 7)
    assert marginal_1["a"][2, 33] == (2 + 6)
    assert marginal_1["a"][2, 44] == (4 + 8)
    assert marginal_1["b"][1, 33] == (9 + 13)
    assert marginal_1["b"][1, 44] == (11 + 15)

    marginal_1 = con_1.marginal("X2", "X3")
    assert all(compare(marginal_1.names, ["X1"]))
    assert all(compare(marginal_1.children_names, ["X4"]))
    assert marginal_1["a"][33] == approx((1 + 5 + 2 + 6) / 36)
    assert marginal_1["a"][44] == approx((3 + 7 + 4 + 8) / 36)
    assert marginal_1["b"][33] == approx((9 + 10 + 13) / 86)
    assert marginal_1["b"][44] == approx((11 + 12 + 15 + 16) / 86)

    marginal_1 = con_1.marginal("X2", "X4")
    assert all(compare(marginal_1.names, ["X1"]))
    assert all(compare(marginal_1.children_names, ["X3"]))
    assert marginal_1["a"][1] == approx((1 + 3 + 5 + 7) / 36)
    assert marginal_1["a"][2] == approx((2 + 4 + 6 + 8) / 36)
    assert marginal_1["b"][1] == approx((9 + 11 + 13 + 15) / 86)
    assert marginal_1["b"][2] == approx((10 + 12 + 16) / 86)

    con_2 = table1.condition_on("X1", "X3")
    with pytest.raises(ValueError):
        con_2.marginal("X1")

    with pytest.raises(ValueError):
        con_2.marginal("X3")

    with pytest.raises(ValueError):
        con_2.marginal("X2", "X4")

    marginal_2 = con_2.marginal("X2")
    assert all(compare(marginal_2.names, ["X1", "X3"]))
    assert all(compare(marginal_2.children_names, ["X4"]))
    assert marginal_2["a", 1][33] == approx((1 + 5) / 16)
    assert marginal_2["a", 1][44] == approx((3 + 7) / 16)
    assert marginal_2["a", 2][44] == approx((4 + 8) / 20)
    assert marginal_2["b", 1][33] == approx((9 + 13) / 48)
    assert marginal_2["b", 2][33] == approx(10 / 38)
    assert marginal_2["b", 2][44] == approx((12 + 16) / 38)

    con_3 = table1.condition_on("X1", "X3", "X4")
    with pytest.raises(ValueError):
        con_3.marginal("X2")
