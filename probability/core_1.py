from abc import ABC  # , abstractmethod
from collections.abc import Mapping, Iterable
from collections import namedtuple
import numpy as np


ColumnsInfo = namedtuple(
    "ColumnsInfo",
    ["indices", "indices_names", "complimnet_indices", "complimnet_names"],
)


class RowKey(tuple):
    def __new__(cls, value):
        try:
            # Using type instead of isinstance is faster
            # This is one of the performance bottleneck
            # Also, for numbers it uses the ducking to
            # catch the exception which is quite fast
            if type(value) is str:
                return tuple.__new__(cls, (value,))
            return tuple.__new__(cls, value)
        except TypeError:  # e.g. key is int
            return tuple.__new__(cls, (value,))


class Column:
    def __init__(self, index, name, table_columns):
        self.index = index
        self.name = name
        self.table_columns = table_columns

    def levels(self):
        keys = [key[self.index] for key in self.table_columns.table.keys()]
        return np.unique(keys)

    def __str__(self):
        return f"{self.name}"

    __repr__ = __str__


class TableColumns(Iterable):
    """Store the details of or more discrete random variables."""

    def __init__(self, names, children_names, table):
        self.names = names
        self.children_names = children_names
        self._columns_ = [Column(index, name, self) for index, name in enumerate(names)]
        self.size = len(self._columns_)
        self.table = table

    def to_key(self, *args, **kwargs):
        total_size = len(args) + len(kwargs.keys())
        if total_size != self.size:
            raise ValueError(
                f"Multi-random variables '{self.names}' can accept {self.size} "
                f"number of levels, {total_size} provided."
            )

        if len(args) == self.size:
            return RowKey(args)

        return_list = [None] * self.size
        for key_name in kwargs:
            if key_name not in self.names:
                raise ValueError(
                    f"The provided name '{key_name}' is not"
                    " defined in multi-random variables (make"
                    f" sure of upper/lower cases too):'{self.names}'"
                )
            # find the index of the name
            index = self.index_of(key_name)
            # fill the return_list in the right place
            return_list[index] = kwargs[key_name]
        # the remaining values are filled in between
        moving_index = 0
        for i, value in enumerate(return_list):
            if value is None:
                return_list[i] = args[moving_index]
                moving_index += 1
        return RowKey(return_list)

    def to_dict_key(self, *args, **kwargs):
        return {
            RowKey(key): value
            for key, value in zip(self.names, self.to_key(*args, **kwargs))
        }

    def named_key(self, key):
        return {name: key[i] for i, name in enumerate(self.names)}

    def __getitem__(self, index):
        """An indexer by position (int) or name (str).

        Args:
            index (int or str): Random variable's name or index.

        Returns:
            DiscreteRV: An instance of DiscreteRV.
        """
        if isinstance(index, int):
            return self._columns_[index]
        elif isinstance(index, str):
            i = self.index_of(index)
            if i > -1:
                return self._columns_[i]
            else:
                raise ValueError(f"Cannot find the column: '{index}'.")
        raise ValueError("The provided index is not 'int' or 'str'.")

    def index_of(self, name):
        """Finds the index of the random variable from its name.

        Args:
            name (str): Name of the random variable.

        Returns:
            int: Index of the random variable.
        """
        for col in self._columns_:
            if col.name == name:
                return col.index

        return -1

    def is_multitable(self):
        return len(self.children_names) > 0

    def __len__(self):
        return self.size

    def __str__(self):
        return "".join([f"{s}, " for s in self.names])

    __repr__ = __str__

    def __contains__(self, name):
        """Check the name of the random variable.

        Args:
            name (str): Name of the random variables.

        Returns:
            [bool]: True or False
        """
        return name in self.names

    def __iter__(self):
        return iter(self._columns_)

    def split_columns(self, *args):
        by_names = args
        for name in by_names:
            if name not in self.names and name not in self.children_names:
                raise ValueError(f"Column name: '{name}' is not defined.")

        indices = [i for i, name in enumerate(self.names) if name in by_names]
        indices_names = [self.names[i] for i in indices]
        # Find the indices of compliment columns (the other ones that
        # are not part of by_names). It can be an empty list
        comp_indices = [i for i in range(self.size) if i not in indices]
        comp_names = [self.names[i] for i in comp_indices]

        return ColumnsInfo(
            indices=indices,
            indices_names=indices_names,
            complimnet_indices=comp_indices,
            complimnet_names=comp_names,
        )

    def contains_child(self, name):
        return name in self.children_names


class Distribution(ABC, Mapping):
    def __init__(self, table):
        self.table = table

    def __len__(self):
        return self.table.__len__()

    def __getitem__(self, key):
        value = self.table[key]
        if value is None:
            return 0

        return value

    def __iter__(self):
        return self.table.__iter__()

    def prob(self, *args, **kwargs):
        key = self.to_key(*args, **kwargs)
        return self.probability(key)

    def probability(self, key):
        """Gets the probability of the random variable, when its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object):
                the value of the random variable.

        Returns:
            float: probability of the random variable.
        """
        return self.__getitem__(key)

    def to_key(self, *args, **kwargs):
        return self.table.columns.to_key(*args, **kwargs)

    def normalise(self):
        """Normalise the distribution."""
        self.table.normalise()

    def to_table(self, sort=False, value_title="Probability"):
        return self.table.to_table(sort, value_title)
