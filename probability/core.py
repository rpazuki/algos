from numbers import Number
from collections.abc import Mapping, Iterable
import numpy as np


class RowKey(tuple):
    def __new__(cls, value):
        try:
            if isinstance(value, str) or isinstance(value, Number):
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
        keys = [key(self.index) for key in self.table_columns.table.keys()]
        return np.unique(keys)


class TableColumns:
    """Store the details of or more discrete random variables."""

    def __init__(self, names, table):
        self.names = names
        self.columns = [Column(index, name, self) for index, name in enumerate(names)]
        self.size = len(self.columns)
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

        super().__init__(key_values)
        self.table_columns = TableColumns(names, self)
        self.names = names

    def __missing__(self, key):
        return None

    def __getitem__(self, args):
        if self.table_columns.size == 1:
            key = self.table_columns.to_key(args)
        else:
            key = self.table_columns.to_key(*args)
        return super().__getitem__(key)

    def _to_2d_array_(self):
        """Convert the distribution ( or the self._counter's
           key:value) to a 2D numpy array where the array
           rows are [[(RV_1, RV_2, ..., RV_n, count)],[...

        Returns:
            numpy ndarray:
                A 2D numpy array that the its last column
                is the counts.
        """
        return np.array([RowKey(k) + (v,) for k, v in self.items()], dtype=np.object)

    def get(self, *args, **kwargs):
        key = self.table_columns.to_key(*args, **kwargs)
        return super().__getitem__(key)

    def to_table(self, sort=False):

        arr = self._to_2d_array_().astype("U")
        arr_len = np.apply_along_axis(lambda row: [len(item) for item in row], 0, arr)
        max_levels_len = np.max(arr_len[:, :-1], axis=0)

        max_freq_len = np.max(arr_len[:, -1])

        def padding(max_len):
            def str_padding(value):
                return "".join([" "] * (max_len - len(str(value))))

            return str_padding

        r_padding = padding(max_freq_len)

        if sort:  # sort by values
            items = reversed(sorted(self.items(), key=lambda item: item[1]))
        else:  # sort by keys
            items = sorted(self.items())

        rows = ""
        header = ""
        horizontal_line = ""
        for i, name in enumerate(self.table_columns.names):
            header += f"|{name}{padding(max_levels_len[i])(name)}"
            horizontal_line += "|" + "".join(["-"] * max_levels_len[i])
        header += "|" + "".join([" "] * max_freq_len) + "|"
        horizontal_line += "|" + "".join(["-"] * max_freq_len) + "|"

        for k, value in items:
            key_str = ""
            for i, k_part in enumerate(k):
                key_str += f"|{padding(max_levels_len[i])(k_part)}{k_part}"
            freq_padding = r_padding(value)
            rows += f"{key_str}|{value}{freq_padding}|\n"

        return f"{header}\n{horizontal_line}\n{rows}"

    def product(self, right):
        if not isinstance(right, Table):
            raise ValueError("The 'right' argument must be a Table.")

        # Find common variables
        # reorder commons based on their order in left_common_indices
        commons = [
            name for name in self.names if name in (set(self.names) & set(right.names))
        ]
        # When there is no common variable, it is just a simple product
        if len(commons) == 0:
            names = np.r_[self.names, right.names]
            return Table(
                {
                    k1 + k2: v1 * v2
                    for k1, v1 in self.items()
                    for k2, v2 in right.items()
                },
                names,
            )
        # In the case that there is one or more common variables,
        # the operation is similar to SQL inner join
        # So, create a lookup for the left table, by using the
        # common variables as key.
        left_common_indices = [
            i for i, name in enumerate(self.names) if name in commons
        ]
        # the order in right must be the same as the left
        # so we reorder the indices base on its left order
        right_common_indices = [
            i
            for name in commons
            for i, name2 in enumerate(right.names)
            if name == name2
        ]
        right_complement_indices = [
            i for i, name in enumerate(right.names) if name not in commons
        ]
        # Methods to split the keys

        def l_comm(key):
            return tuple([key[i] for i in left_common_indices])

        def r_comm(key):
            return tuple([key[i] for i in right_common_indices])

        def r_comp(key):
            return tuple([key[i] for i in right_complement_indices])

        # left and right tables lookup
        # left : (key:value) == (common_key: (left_key, left_value))
        left_lookup = {}
        for k, value in self.items():
            comm = l_comm(k)
            if comm in left_lookup:
                left_lookup[comm] += [(k, value)]
            else:
                left_lookup[comm] = [(k, value)]
        # right : (key:value) == (common_key: (right_compliment_key, right_value))
        right_lookup = {}
        for k, value in right.items():
            comm = r_comm(k)
            if comm in right_lookup:
                right_lookup[comm] += [(r_comp(k), value)]
            else:
                right_lookup[comm] = [(r_comp(k), value)]
        # The inner join happens over keys of two dictionaries (left_lookup and
        # right_lookup).
        prodcut_dict = {}
        for comm, l_values in left_lookup.items():
            if comm not in right_lookup:
                continue
            for left_key, left_value in l_values:
                for right_comp, right_value in right_lookup[comm]:
                    # prodcut_dict values must be multiplied.
                    # prodcut_dict keys are the combination: (left, right_compliment).
                    prodcut_dict[RowKey(left_key) + RowKey(right_comp)] = (
                        left_value * right_value
                    )
        # names are the combination of [left_names, right_compelements_names]
        combined_names = np.r_[
            [name for name in self.names],
            [name for name in right.names if name not in commons],
        ]
        # TODO: Table convert the key to RowKey for second time
        return Table(prodcut_dict, combined_names)

    def __mul__(self, right):
        if not isinstance(right, Table):
            raise ValueError("The 'right' argument must be a 'Table'.")

        return self.product(right)

    def __rmul__(self, left):
        if not isinstance(left, Table):
            raise ValueError("The 'right' argument must be a DiscreteDistribution.")

        return left.product(self)
