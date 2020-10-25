from probability.core import RowKey


def test_conversion_key():
    key_1 = RowKey(1)
    assert key_1 == (1,)

    key_1 = RowKey(1.0)
    assert key_1 == (1.0,)

    key_1 = RowKey("A")
    assert key_1 == ("A",)

    key_1 = RowKey("ABC")
    assert key_1 == ("ABC",)

    key_1 = RowKey((1,))
    assert key_1 == (1,)

    key_1 = RowKey((1, 2))
    assert key_1 == (1, 2)

    key_1 = RowKey((1, 2, 3))
    assert key_1 == (1, 2, 3)

    key_1 = RowKey([1])
    assert key_1 == (1,)

    key_1 = RowKey([1, 2])
    assert key_1 == (1, 2)

    key_1 = RowKey([1, 2, 3])
    assert key_1 == (1, 2, 3)
