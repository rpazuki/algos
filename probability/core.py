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


class Column:
    def __init__(self, index, name):
        self.index = index
        self.name = name


class TableColumns:
    """Store the details of or more discrete random variables."""

    def __init__(self, names):
        self.names = names
        self.columns = [Column(index, name) for index, name in enumerate(names)]
        self.size = len(self.columns)

    def to_key(self, *args, **kwargs):
        total_size = len(args) + len(kwargs.keys())
        if total_size != self.size:
            raise ValueError(
                f"Multi-random variables '{self.names}' can accept {self.size} "
                f"number of levels, {total_size} provided."
            )

        if len(args) == self.size and self.size > 1:
            return RowKey(args)
        elif len(args) == self.size:  # self.size == 1
            return args[0]

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
        if self.size == 1:
            return return_list[0]
        else:
            return RowKey(return_list)

    def to_dict_key(self, *args, **kwargs):
        return {
            RowKey(key): value
            for key, value in zip(self.names, self.to_key(*args, **kwargs))
        }

    def __getitem__(self, index):
        """An indexer by position (int) or name (str).

        Args:
            index (int or str): Random variable's name or index.

        Returns:
            DiscreteRV: An instance of DiscreteRV.
        """
        if isinstance(index, int):
            return self.columns[index]
        elif isinstance(index, str):
            i = self.index_of(index)
            if i > -1:
                return self.columns[i]
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
        for col in self.columns:
            if col.name == name:
                return col.index

        return -1

    def __len__(self):
        return self.size

    def __str__(self):
        return "".join([f"{s}\n" for s in self.names])

    __repr__ = __str__

    def __contains__(self, name):
        """Check the name of the random variable.

        Args:
            name (str): Name of the random variables.

        Returns:
            [bool]: True or False
        """
        return name in self.names


class Table(dict):
    def __init__(self, rows, names=None):

        if isinstance(rows, Mapping):
            key_values = [(RowKey(k), value) for k, value in rows.items()]
        elif isinstance(rows, Iterable):
            key_values = [(RowKey(k), value) for k, value in rows]
        else:
            raise ValueError("Table expect rows as Mapping/Iterable")

        row_sample = key_values[0][0]

        if names is None:
            names = [f"X{i+1}" for i, _ in enumerate(row_sample)]

        if len(names) != len(row_sample):
            raise ValueError("The length of column names and columns are not the same.")

        self.headers = TableColumns(names)
        super().__init__(key_values)

    def __missing__(self, key):
        return None

    def __getitem__(self, key):
        return super().__getitem__(RowKey(key))
