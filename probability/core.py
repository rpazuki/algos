from collections.abc import Mapping, Iterable
from itertools import groupby
from operator import itemgetter
import numpy as np


class RowKey(tuple):
    def __new__(cls, value):
        try:
            # if isinstance(value, str) or isinstance(value, Number):
            # if isinstance(value, str):
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


class TableColumns(Iterable):
    """Store the details of or more discrete random variables."""

    def __init__(self, names, table):
        self.names = names
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

    def __iter__(self):
        return iter(self._columns_)


class Table(dict):
    def __init__(self, rows, names=None, _internal_=False):

        if _internal_:
            key_values = rows
            self._row_sample_ = next(iter(rows))
        else:
            if isinstance(rows, Mapping):
                key_values = [(RowKey(k), value) for k, value in rows.items()]
            elif isinstance(rows, Iterable):
                key_values = [(RowKey(k), value) for k, value in rows]
            else:
                raise ValueError("Table expect rows as Mapping/Iterable")

            self._row_sample_ = key_values[0][0]

        if names is None:
            names = [f"X{i+1}" for i, _ in enumerate(self._row_sample_)]

        if len(names) != len(self._row_sample_):
            raise ValueError("The length of column names and columns are not the same.")

        super().__init__(key_values)
        self.columns = TableColumns(names, self)
        self.names = names

    def __missing__(self, key):
        return None

    def __getitem__(self, args):
        if self.columns.size == 1:
            key = self.columns.to_key(args)
        else:
            key = self.columns.to_key(*args)
        return super().__getitem__(key)

    def _check_keys_consistencies_(self):
        # We suppose each column is positioned
        # in a fix place of the n-tuple.
        # Therefore, the levels of the column can be
        # found by iterating over each tuple's item
        # Convert each features line to tuple
        first_row_types = [type(item) for item in self._row_sample_]
        for row in self.keys():
            # compair length
            if len(row) != self.columns.size:
                raise ValueError("The length of the 'factors' are not consistence.")
            # compair row's elements type
            comparisions = [
                isinstance(element, type_1)
                for element, type_1 in zip(row, first_row_types)
            ]
            if not all(comparisions):
                raise ValueError("The types of the 'factors' are not consistence.")

    def _to_2d_array_(self):
        """Convert the distribution ( or the self._counter's
           key:value) to a 2D numpy array where the array
           rows are [[(RV_1, RV_2, ..., RV_n, count)],[...

        Returns:
            numpy ndarray:
                A 2D numpy array that the its last column
                is the counts.
        """
        return np.array([k + (v,) for k, v in self.items()], dtype=np.object)

    def _split_matrix_(self, array, indices=None):
        """Convert the 2D array of the Table
           to a generator of 2-elements tuple like
           ((col_1, col_2, ..., col_n), value)

           If the indices is provided, the tuple is
           like
           ((col_1, ..., col_n), (c_col_1, ..., c_col_m), value)
           where col_i is from indices and c_col_j is
           from its compliment

        Args:
            array (numpy ndarray):
                A 2D array of Table
            indices (list, optional):
                list of indices to filter the column.
                Defaults to None.
        """

        # divide the 2d array's rows to a tuple of
        # columns (row[indices])
        # and value row[-1]
        if indices is None:
            return ((RowKey(row[:-1]), row[-1]) for row in array)

        # Find the indices of compliment columns (the other ones that
        # are not part of conditioning)
        comp_indices = np.array(
            [i for i in range(self.columns.size) if i not in indices]
        )
        # divide the 2d array's rows to a tuple of columns,
        # (row[indices]), compliment columns (row[comp_indices])
        # and value row[-1]
        return (
            (RowKey(row[indices]), RowKey(row[comp_indices]), row[-1]) for row in array
        )

    def _product_(self, right):
        """Multiply two Tables.

        Args:
            right ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
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
            return (
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
                    prodcut_dict[left_key + right_comp] = left_value * right_value

        # names are the combination of [left_names, right_compelements_names]
        combined_names = np.r_[
            [name for name in self.names],
            [name for name in right.names if name not in commons],
        ]
        return (prodcut_dict, combined_names)

    def _group_by_(self, *args):
        """group by the Table over a set of columns.

        Args:
            args (list):
                List of column names to marginalised.

        Raises:
            ValueError:
                Raises when one of the column names is
                not defined.
                Or raises when requested for all column names.

        Returns:
            Table: (rows, names).
        """
        by_names = args
        for name in by_names:
            if name not in self.names:
                raise ValueError(f"Column name: '{name}'' is not defined.")

        if len(by_names) == self.columns.size:
            raise ValueError("Cannot marginalize on all column names.")

        indices = [i for i, name in enumerate(self.names) if name in by_names]
        # Find the indices of compliment columns (the other ones that
        # are not part of conditioning)
        comp_indices = np.array(
            [i for i in range(self.columns.size) if i not in indices]
        )
        # Convert the key:values to 2D numpy array
        # the array rows are (row, value)
        arr = self._to_2d_array_()
        # filter the compliment random variables by columns
        filtered_arr = np.c_[arr[:, comp_indices], arr[:, -1]]
        # divide the 2d array's rows to a tuple of
        # compliment columns (row[comp_indices])
        # and count row[-1]
        arr_gen = self._split_matrix_(filtered_arr)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of compliment columns (index zero in itemgetter)
        sorted_arr = sorted(arr_gen, key=itemgetter(0))
        # since the values in each 'group' are
        # (compliment columns, value)
        # here we group by 'compliment columns' and apply
        # the sum on the value. Then the dictionary of
        # compliment columns:op_of_values
        # is an acceptable argument for Table
        grouped_arr = {
            k: sum([item[1] for item in g])
            for k, g in groupby(sorted_arr, key=itemgetter(0))
        }
        return (grouped_arr, [self.names[i] for i in comp_indices])

    def _group_on_(self, *args):
        """Creates the conditional based on
           the provided names of columns.

        Args:
            args (list):
                List of names of provided random
                variables.

        Raises:
            ValueError:
                If the provided RV names do not exist
                in the distribution.

        Returns:
            (row, names)
        """
        on_names = args
        for name in on_names:
            if name not in self.names:
                raise ValueError(f"Column name '{name}' is not defined.")

        if len(on_names) == self.columns.size:
            raise ValueError("Cannot condition on all columns.")

        if self.columns.size == 1:
            raise TypeError("This is a single column Table.")

        indices = [i for i, name in enumerate(self.names) if name in on_names]
        # Find the indices of compliment columns (the other ones that
        # are not part of conditioning)
        comp_indices = np.array(
            [i for i in range(self.columns.size) if i not in indices]
        )
        # Convert the key:value to 2D numpy array
        # the array rows are (rows, value)
        arr = self._to_2d_array_()
        # divide the 2d array's rows to a tuple of columns,
        # (row[indices]), compliment columns (row[comp_indices])
        # and values row[-1]
        arr_gen = self._split_matrix_(arr, indices)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of columns (index zero in itemgetter)
        # And since later we will call the group by on group,
        # for each key we do the inner sort too (index one in itemgetter)
        sorted_arr = sorted(arr_gen, key=itemgetter(0, 1))
        # This method convert a group to a dictionary

        def make_dict(group):
            # since the values in 'group' argument are
            # (columns, compliment columns, value)
            # here we group by 'compliment columns' and sum
            # the values.
            return {
                k: sum([item[2] for item in g2])
                for k, g2 in groupby(group, key=itemgetter(1))
            }

        # For each group (belongs a unique values), we create
        # a dictionary in a dictionary comprehension
        grouped_arr = {
            k: make_dict(g) for k, g in groupby(sorted_arr, key=itemgetter(0))
        }
        # The above dictionary is dictionary of dictionaries
        # # the first set of names is for parent dictionary
        # and the second set is for children
        return (
            grouped_arr,
            [self.names[i] for i in indices],
            [self.names[i] for i in comp_indices],
        )

    def get(self, *args, **kwargs):
        key = self.columns.to_key(*args, **kwargs)
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
        for i, name in enumerate(self.names):
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

    def marginal(self, *args):
        """Marginalize the Table over a set of columns.

        Args:
            args (list):
                List of column names to marginalised.

        Raises:
            ValueError:
                Raises when one of the column names is
                not defined.
                Or raises when requested for all column names.

        Returns:
            Table: A new marginalised Table.
        """
        (rows, names) = self._group_by_(*args)
        return Table(rows, names, _internal_=True)

    def reduce(self, **kwargs):
        """Reduce the Table by one or more columns.

        Args:
            kwargs (dict):
                A dictionary that its 'key' is the name
                of the column and its 'value'
                is the value that must be reduced by.

        Raises:
            ValueError:
                If the provided names do not exist in the Table.

        Returns:
            [Table]: A reduce Table.
        """
        by_names = kwargs
        for name in by_names:
            if name not in self.names:
                raise ValueError(f"The column name '{name}'' is" f" not defined.")

        indices = [self.columns.index_of(col) for col, _ in by_names.items()]
        values = np.array([value for _, value in by_names.items()], dtype=np.object)
        #
        # Convert the key:values to 2D numpy array
        # the array rows are (keys, value)
        arr_counter = self._to_2d_array_()
        # Find the indices of compliment columns (the other ones that
        # are not part of reduce)
        compliment_indices = [i for i in range(self.columns.size) if i not in indices]
        # filter the 2d array rows by provided values of the reduce
        # conditioned_arr is a boolean one, and filtering happens
        # in the second line
        conditioned_arr = np.all(arr_counter[:, indices] == values, axis=1)
        sliced_arr = arr_counter[conditioned_arr, :]
        # filter the 2d array columns (the compliment columns)
        # plus the value column (which is the last column)
        sliced_arr = sliced_arr[:, compliment_indices + [-1]]
        # divide the 2d array's rows to a tuple of columns
        # and value
        # So, we make a generator that divide the rows to the tuple of
        # columns (tuple(row[:-1]) and value (row[-1])
        arr_gen = self._split_matrix_(sliced_arr)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of column (index zero in itemgetter)
        sorted_slice_arr = sorted(arr_gen, key=itemgetter(0))
        # group by the filtered columns (compliment
        # columns) and sum the value per key
        # Note that the 'itemgetter' read the first index which
        # is the tuple of compliment columns
        slice_dist = {
            k: sum([item[1] for item in g])
            for k, g in groupby(sorted_slice_arr, key=itemgetter(0))
        }
        # Since we have a dictionary of (compliment columns: values),
        # it is easy to create a Table.
        # Obviously, the names of these columns must
        # be the same as compliment columns, which we selected
        # from self.rvs.names
        return Table(slice_dist, [self.names[i] for i in compliment_indices])

    def add(self, that):
        """Combines two FrequencyTable and return
        a new one. All the frequencies are sum together.
        This is not a mathematical sum.
        """
        if not isinstance(that, Table):
            raise ValueError("Table can only adds to Table.")

        for name in self.names:
            if name not in that.names:
                raise ValueError(
                    "Two adding Table do not have the same random variables."
                )
        this_copy = self.copy()
        for key in that.keys():
            if key in this_copy:
                this_copy[key] += that[key]
            else:
                this_copy[key] = that[key]

        return Table(this_copy, names=self.names)

    def __mul__(self, right):
        if not isinstance(right, Table):
            raise ValueError("The 'right' argument must be a 'Table'.")

        (rows, names) = self._product_(right)
        return Table(rows, names, _internal_=True)

    def __rmul__(self, left):
        if not isinstance(left, Table):
            raise ValueError("The 'right' argument must be a DiscreteDistribution.")

        (rows, names) = left._product_(self)
        return Table(rows, names, _internal_=True)

    def __add__(self, right):
        return self.add(right)
