import pytest
from probability.distributions import DiscreteRV
from tests.helpers import compare_in


def test_create_discreterv():
    random_v = DiscreteRV("X1", 1)
    assert random_v.name == "X1"
    assert random_v.is_numeric

    random_v = DiscreteRV("X1", (1,))
    assert not random_v.is_numeric

    random_v = DiscreteRV("X1", (1, 1))
    assert not random_v.is_numeric

    random_v = DiscreteRV("X1", "1")
    assert not random_v.is_numeric

    random_v = DiscreteRV("X1", "111")
    assert not random_v.is_numeric


def test_len_discreterv():
    # It must be always equal to one
    random_v = DiscreteRV("X1", 1)
    assert len(random_v) == 1
    assert random_v.size == 1

    random_v = DiscreteRV("X1", (1,))
    assert len(random_v) == 1
    assert random_v.size == 1

    random_v = DiscreteRV("X1", (1, 1))
    assert len(random_v) == 1
    assert random_v.size == 1

    random_v = DiscreteRV("X1", "1")
    assert len(random_v) == 1
    assert random_v.size == 1

    random_v = DiscreteRV("X1", "111")
    assert len(random_v) == 1
    assert random_v.size == 1


def test_to_key_exceptions_discreterv():
    # wrong name
    random_v = DiscreteRV("X1", 1)
    with pytest.raises(ValueError):
        random_v.to_key(x2=2)

    random_v = DiscreteRV("X1", (1,))
    # wrong name
    with pytest.raises(ValueError):
        random_v.to_key(x1=(1,))

    # wrong length
    with pytest.raises(ValueError):
        random_v.to_key(1, 2)

    with pytest.raises(ValueError):
        random_v.to_key()

    with pytest.raises(ValueError):
        random_v.to_key(1, x1=2)

    with pytest.raises(ValueError):
        random_v.to_key(1, x2=2)

    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=2)

    random_v = DiscreteRV("X1", (1,))

    with pytest.raises(ValueError):
        random_v.to_key(1, 2)

    with pytest.raises(ValueError):
        random_v.to_key()

    with pytest.raises(ValueError):
        random_v.to_key(1, x1=2)

    with pytest.raises(ValueError):
        random_v.to_key(1, x2=2)

    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=2)

    random_v = DiscreteRV("X1", (1, 1))

    with pytest.raises(ValueError):
        random_v.to_key(1, 2, 3)

    with pytest.raises(ValueError):
        random_v.to_key(1, 2)

    with pytest.raises(ValueError):
        random_v.to_key()

    with pytest.raises(ValueError):
        random_v.to_key(1, 3, x1=2)

    with pytest.raises(ValueError):
        random_v.to_key(1, x2=2)

    with pytest.raises(ValueError):
        random_v.to_key(x1=1, x2=2)


def test_to_key_discreterv():
    random_v = DiscreteRV("X1", 1)
    assert random_v.to_key(2) == 2
    assert random_v.to_key(X1=2) == 2

    random_v = DiscreteRV("X1", (1,))
    assert random_v.to_key((1,)) == (1,)
    assert random_v.to_key(X1=(1,)) == (1,)

    random_v = DiscreteRV("X1", (1, 1))
    assert random_v.to_key((1, 1)) == (1, 1)
    assert random_v.to_key(X1=(2, 2)) == (2, 2)

    random_v = DiscreteRV("X1", "1")
    assert random_v.to_key("1") == "1"
    assert random_v.to_key(X1="1") == "1"

    random_v = DiscreteRV("X1", "111")
    assert random_v.to_key("111") == "111"
    assert random_v.to_key(X1="222") == "222"


def test_to_dict_key_discreterv():
    random_v = DiscreteRV("X1", 1)
    assert random_v.to_dict_key(2) == {"X1": 2}
    assert random_v.to_dict_key(X1=2) == {"X1": 2}

    random_v = DiscreteRV("X1", (1,))
    assert random_v.to_dict_key((1,)) == {"X1": (1,)}
    assert random_v.to_dict_key(X1=(1,)) == {"X1": (1,)}

    random_v = DiscreteRV("X1", (1, 1))
    assert random_v.to_dict_key((1, 1)) == {"X1": (1, 1)}
    assert random_v.to_dict_key(X1=(2, 2)) == {"X1": (2, 2)}

    random_v = DiscreteRV("X1", "1")
    assert random_v.to_dict_key("1") == {"X1": "1"}
    assert random_v.to_dict_key(X1="1") == {"X1": "1"}

    random_v = DiscreteRV("X1", "111")
    assert random_v.to_dict_key("111") == {"X1": "111"}
    assert random_v.to_dict_key(X1="222") == {"X1": "222"}
