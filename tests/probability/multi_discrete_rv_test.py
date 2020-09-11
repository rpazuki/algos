import pytest
from probability.distributions import MultiDiscreteRV


def test_len_multidiscreterv():
    random_v = MultiDiscreteRV([1, 2, 3, 4], ["X1"])
    assert len(random_v) == 1
    assert random_v.size == 1

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert len(random_v) == 2
    assert random_v.size == 2

    random_v = MultiDiscreteRV((1, "aaa", "w1"))
    assert len(random_v) == 3
    assert random_v.size == 3

    random_v = MultiDiscreteRV((1, 2, 4, 2), ["X1", "X2", "X3", "X4"])
    assert len(random_v) == 4
    assert random_v.size == 4


def test_index_of_multidiscreterv():
    random_v = MultiDiscreteRV(1, ["X1"])
    assert random_v.index_of("X1") == 0
    assert random_v.index_of("Y1") == -1

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert random_v.index_of("X1") == 0
    assert random_v.index_of("X2") == 1
    assert random_v.index_of("Y1") == -1

    random_v = MultiDiscreteRV((1, "aaa", "w1"), ["X1", "X2", "X3"])
    assert random_v.index_of("X1") == 0
    assert random_v.index_of("X2") == 1
    assert random_v.index_of("X3") == 2
    assert random_v.index_of("Y1") == -1

    random_v = MultiDiscreteRV(
        (1, 2, 4, 2),
        ["X1", "X2", "X3", "X4"],
    )
    assert random_v.index_of("X1") == 0
    assert random_v.index_of("X2") == 1
    assert random_v.index_of("X3") == 2
    assert random_v.index_of("X4") == 3
    assert random_v.index_of("Y1") == -1


def test_indexer_multidiscreterv():
    random_v = MultiDiscreteRV(1, ["X1"])
    assert random_v[0].name == "X1"
    assert random_v["X1"].name == "X1"
    with pytest.raises(ValueError):
        assert random_v[(1,)] == "X1"

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert random_v[0].name == "X1"
    assert random_v["X1"].name == "X1"
    assert random_v[1].name == "X2"
    assert random_v["X2"].name == "X2"

    random_v = MultiDiscreteRV(
        (1, "aaa", "w1"),
        ["X1", "X2", "X3"],
    )
    assert random_v[0].name == "X1"
    assert random_v["X1"].name == "X1"
    assert random_v[1].name == "X2"
    assert random_v["X2"].name == "X2"
    assert random_v[2].name == "X3"
    assert random_v["X3"].name == "X3"

    random_v = MultiDiscreteRV(
        (1, 2, 4, 2),
        ["X1", "X2", "X3", "X4"],
    )
    assert random_v[0].name == "X1"
    assert random_v["X1"].name == "X1"
    assert random_v[1].name == "X2"
    assert random_v["X2"].name == "X2"
    assert random_v[2].name == "X3"
    assert random_v["X3"].name == "X3"
    assert random_v[3].name == "X4"
    assert random_v["X4"].name == "X4"


def test_contains_multidiscreterv():
    random_v = MultiDiscreteRV(1)
    assert "X1" in random_v
    assert "Y1" not in random_v

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert "X1" in random_v
    assert "X2" in random_v
    assert "Y1" not in random_v

    random_v = MultiDiscreteRV(
        (1, "aaa", "w1"),
        ["X1", "X2", "X3"],
    )
    assert "X1" in random_v
    assert "X2" in random_v
    assert "X3" in random_v
    assert "Y1" not in random_v

    random_v = MultiDiscreteRV(
        (1, 2, 4, 2),
        ["X1", "X2", "X3", "X4"],
    )
    assert "X1" in random_v
    assert "X2" in random_v
    assert "X3" in random_v
    assert "X4" in random_v
    assert "Y1" not in random_v


def test_to_key_exceptions_multidiscreterv():
    # wrong name
    random_v = MultiDiscreteRV(1, ["X1"])
    with pytest.raises(ValueError):
        random_v.to_key(x2=2)

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x3=2)
    with pytest.raises(ValueError):
        random_v.to_key(x3=1, x1=2)

    random_v = MultiDiscreteRV(
        ("a", 1, "w1"),
        ["X1", "X2", "X3"],
    )
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=3, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x4=2, x2=3)
    with pytest.raises(ValueError):
        random_v.to_key(x4=2, x1=1, x2=3)

    random_v = MultiDiscreteRV(
        (1, 2, 4, 2),
        ["X1", "X2", "X3", "X4"],
    )
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=3, x3=2, x5=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=2, x5=1, x3=3)
    with pytest.raises(ValueError):
        random_v.to_key(x1=2, x5=1, x2=1, x3=3)
    with pytest.raises(ValueError):
        random_v.to_key(x5=1, x1=2, x2=1, x3=3)

    # wrong length
    random_v = MultiDiscreteRV(1, ["X1"])
    with pytest.raises(ValueError):
        random_v.to_key()
    with pytest.raises(ValueError):
        random_v.to_key(1, 2)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1)

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    with pytest.raises(ValueError):
        random_v.to_key()
    with pytest.raises(ValueError):
        random_v.to_key(1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1, x3=3)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1, x3=3)

    random_v = MultiDiscreteRV(
        ("a", 1, "w1"),
        ["X1", "X2", "X3"],
    )
    with pytest.raises(ValueError):
        random_v.to_key()
    with pytest.raises(ValueError):
        random_v.to_key(1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 4, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1, x3=1, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1, x3=1, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1, x3=1, x4=2)

    random_v = MultiDiscreteRV(
        (1, 2, 4, 2),
        ["X1", "X2", "X3", "X4"],
    )
    with pytest.raises(ValueError):
        random_v.to_key()
    with pytest.raises(ValueError):
        random_v.to_key(1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4, 5)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4, 5, 6)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, 4, 5, x1=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, x1=1, x2=1)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1, x3=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1, x3=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, x1=1, x2=1, x3=1)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1, x3=1, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1, x3=1, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, x1=1, x2=1, x3=1, x4=2)
    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=1, x3=1, x4=2, x5=5)
    with pytest.raises(ValueError):
        random_v.to_key(1, x1=1, x2=1, x3=1, x4=2, x5=5)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, x1=1, x2=1, x3=1, x4=2, x5=5)
    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3, x1=1, x2=1, x3=1, x4=2, x5=5)


def test_to_key_multidiscreterv():
    random_v = MultiDiscreteRV(1, ["X1"])
    assert random_v.to_key(2) == 2
    assert random_v.to_key(X1=2) == 2

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert random_v.to_key(1, 2) == (1, 2)
    assert random_v.to_key(1, X2=2) == (1, 2)
    assert random_v.to_key(2, X1=1) == (1, 2)
    assert random_v.to_key(X1=1, X2=2) == (1, 2)
    assert random_v.to_key(X2=2, X1=1) == (1, 2)

    random_v = MultiDiscreteRV(
        ("a", 1, "w1"),
        ["X1", "X2", "X3"],
    )
    assert random_v.to_key("a", 1, "w1") == ("a", 1, "w1")
    assert random_v.to_key("a", 1, X3="w1") == ("a", 1, "w1")
    assert random_v.to_key("a", "w1", X2=1) == ("a", 1, "w1")
    assert random_v.to_key(1, "w1", X1="a") == ("a", 1, "w1")
    assert random_v.to_key("a", X2=1, X3="w1") == ("a", 1, "w1")
    assert random_v.to_key(1, X1="a", X3="w1") == ("a", 1, "w1")
    assert random_v.to_key("w1", X1="a", X2=1) == ("a", 1, "w1")
    assert random_v.to_key("a", X3="w1", X2=1) == ("a", 1, "w1")
    assert random_v.to_key(1, X3="w1", X1="a") == ("a", 1, "w1")
    assert random_v.to_key("w1", X2=1, X1="a") == ("a", 1, "w1")
    assert random_v.to_key(X1="a", X2=1, X3="w1") == ("a", 1, "w1")
    assert random_v.to_key(X1="a", X3="w1", X2=1) == ("a", 1, "w1")
    assert random_v.to_key(X2=1, X1="a", X3="w1") == ("a", 1, "w1")
    assert random_v.to_key(X2=1, X3="w1", X1="a") == ("a", 1, "w1")
    assert random_v.to_key(X3="w1", X1="a", X2=1) == ("a", 1, "w1")
    assert random_v.to_key(X3="w1", X2=1, X1="a") == ("a", 1, "w1")

    random_v = MultiDiscreteRV(
        ("a", 1, "w1", 4),
        ["X1", "X2", "X3", "X4"],
    )
    assert random_v.to_key("a", 1, "w1", 4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 1, "w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 1, 4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", "w1", 4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, "w1", 4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 1, X3="w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 1, X4=4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", "w1", X2=1, X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", "w1", X4=4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, "w1", X1="a", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, "w1", X4=4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 4, X2=1, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", 4, X3="w1", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, 4, X1="a", X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(1, 4, X3="w1", X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", 4, X1="a", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", 4, X2=1, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key("a", X2=1, X3="w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", X3="w1", X2=1, X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", X3="w1", X4=4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key("a", X2=1, X4=4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(1, X1="a", X3="w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, X3="w1", X1="a", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(1, X3="w1", X4=4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(1, X1="a", X4=4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", X1="a", X2=1, X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", X2=1, X1="a", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", X2=1, X4=4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key("w1", X1="a", X4=4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(4, X1="a", X2=1, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(4, X2=1, X1="a", X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(4, X2=1, X3="w1", X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(4, X1="a", X3="w1", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X1="a", X2=1, X3="w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X1="a", X2=1, X4=4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(X1="a", X3="w1", X2=1, X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X1="a", X3="w1", X4=4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X1="a", X3="w1", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X1="a", X4=4, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X3="w1", X1="a", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X3="w1", X4=4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X4=4, X3="w1", X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(X2=1, X4=4, X1="a", X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X1="a", X2=1, X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X1="a", X4=4, X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X2=1, X1="a", X4=4) == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X2=1, X4=4, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X4=4, X1="a", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X3="w1", X4=4, X2=1, X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X1="a", X2=1, X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X1="a", X3="w1", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X2=1, X1="a", X3="w1") == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X2=1, X3="w1", X1="a") == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X3="w1", X1="a", X2=1) == ("a", 1, "w1", 4)
    assert random_v.to_key(X4=4, X3="w1", X2=1, X1="a") == ("a", 1, "w1", 4)


def test_to_dict_key_multidiscreterv():
    random_v = MultiDiscreteRV(1, ["X1"])
    # assert random_v.to_dict_key(2) == {"X1": 2}
    # assert random_v.to_dict_key(X1=2) == {"X1": 2}

    random_v = MultiDiscreteRV((1, 2), ["X1", "X2"])
    assert random_v.to_dict_key(1, 2) == {"X1": 1, "X2": 2}
    assert random_v.to_dict_key(1, X2=2) == {"X1": 1, "X2": 2}
    assert random_v.to_dict_key(X1=1, X2=2) == {"X1": 1, "X2": 2}

    random_v = MultiDiscreteRV(
        ("a", 1, "w1"),
        ["X1", "X2", "X3"],
    )
    assert random_v.to_dict_key("a", 1, "w1") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key("a", 1, X3="w1") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key("a", X2=1, X3="w1") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key(1, X1="a", X3="w1") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key(1, X3="w1", X1="a") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key("w1", X2=1, X1="a") == {"X1": "a", "X2": 1, "X3": "w1"}
    assert random_v.to_dict_key(X1="a", X2=1, X3="w1") == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
    }
    assert random_v.to_dict_key(X3="w1", X1="a", X2=1) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
    }

    random_v = MultiDiscreteRV(
        ("a", 1, "w1", 4),
        ["X1", "X2", "X3", "X4"],
    )
    assert random_v.to_dict_key("a", 1, "w1", 4) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
    assert random_v.to_dict_key("a", 1, "w1", X4=4) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
    assert random_v.to_dict_key("a", 1, X3="w1", X4=4) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
    assert random_v.to_dict_key("a", X2=1, X3="w1", X4=4) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
    assert random_v.to_dict_key(X1="a", X2=1, X3="w1", X4=4) == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
    assert random_v.to_dict_key(X2=1, X1="a", X4=4, X3="w1") == {
        "X1": "a",
        "X2": 1,
        "X3": "w1",
        "X4": 4,
    }
