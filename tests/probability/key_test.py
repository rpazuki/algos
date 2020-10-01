from probability import Key


def test_conversion_key():
    key_1 = Key(1)
    assert key_1 == (1,)

    key_1 = Key(1.0)
    assert key_1 == (1.0,)

    key_1 = Key("A")
    assert key_1 == ("A",)

    key_1 = Key("ABC")
    assert key_1 == ("ABC",)

    key_1 = Key((1,))
    assert key_1 == (1,)

    key_1 = Key((1, 2))
    assert key_1 == (1, 2)

    key_1 = Key((1, 2, 3))
    assert key_1 == (1, 2, 3)

    key_1 = Key([1])
    assert key_1 == (1,)

    key_1 = Key([1, 2])
    assert key_1 == (1, 2)

    key_1 = Key([1, 2, 3])
    assert key_1 == (1, 2, 3)
