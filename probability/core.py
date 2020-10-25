from numbers import Number
from collections.abc import Mapping, Iterable


class RowKey(tuple):
    def __new__(cls, value):
        try:
            if isinstance(value, str) or isinstance(value, Number):
                return tuple.__new__(cls, (value,))
            return tuple.__new__(cls, value)
        except TypeError:  # e.g. key is int
            return tuple.__new__(cls, (value,))


class Table(dict):
    def __init__(self, *args, **kwarg):

        if len(args) > 1:
            raise TypeError("dict expected at most 1 argument, got 2")

        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Mapping):
                super().__init__({RowKey(k): value for k, value in arg.items()})
            elif isinstance(arg, Iterable):
                super().__init__({RowKey(k): value for k, value in arg})
            else:
                raise ValueError("Table expect Mapping/Iterable as positional argument")
        else:
            key_value_args = {RowKey(k): value for k, value in kwarg.items()}
            super().__init__(key_value_args)

    def __missing__(self, key):
        return None

    def __getitem__(self, key):
        return super().__getitem__(RowKey(key))
