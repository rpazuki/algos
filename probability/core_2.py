from collections import ChainMap
from collections.abc import Mapping, Iterable
from itertools import groupby
from operator import itemgetter
import numpy as np
from probability import RowKey
from probability import TableColumns

# from probability.core_1 import RowKey
# from probability.core_1 import TableColumns


def to_dict(groupby_index, value_index):
    def make_dict(sorted_items):
        # It groups the sorted item based on
        # the element as groupby_index
        # and then sum the values at value_index
        return {
            k: sum([item[value_index] for item in g2])
            for k, g2 in groupby(sorted_items, key=itemgetter(groupby_index))
        }

    return make_dict


class Table(dict):
    def __init__(self, rows, names=None, _internal_=False, _children_names_=None):

        if _internal_:
            # rows are dictionary for internal calls
            key_values = rows
            try:
                self._row_sample_ = next(iter(rows))
            except StopIteration:
                # Rows are empty
                super().__init__(key_values)
                self._row_sample_ = None
                self.names = names
                if _children_names_ is None:
                    self.children_names = []
                    self.columns = TableColumns(
                        names=names, children_names=[], table=self
                    )
                else:
                    self.children_names = _children_names_
                    self.columns = TableColumns(
                        names=names, children_names=_children_names_, table=self
                    )
                return
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

        self.names = names
        value_sample = super().__getitem__(self._row_sample_)
        if isinstance(value_sample, Table):
            self.columns = TableColumns(
                names=names, children_names=value_sample.names, table=self
            )
            self.children_names = value_sample.names
        else:
            if _children_names_ is None:
                self.children_names = []
                self.columns = TableColumns(names=names, children_names=[], table=self)
            else:
                self.children_names = _children_names_
                self.columns = TableColumns(
                    names=names, children_names=_children_names_, table=self
                )

    def __missing__(self, key):
        return None

    def __getitem__(self, args):
        """Override the dict by converting the
        comma separated arguments to RowKey
        """
        # This is faster than isinstance
        # We are sure there is not any inheritance
        # to deal with
        if type(args) is RowKey:
            return super().__getitem__(args)

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

    def to_2d_array(self):
        """Convert the distribution ( or the self._counter's
           key:value) to a 2D numpy array where the array
           rows are [[(RV_1, RV_2, ..., RV_n, count)],[...

        Returns:
            numpy ndarray:
                A 2D numpy array that the its last column
                is the counts.
        """
        return np.array([k + (v,) for k, v in self.items()], dtype=np.object)

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
            self.names,
            [name for name in right.names if name not in commons],
        ]
        return (prodcut_dict, combined_names)

    def marginal(self, *args, normalise=True):
        """Marginal of (group by) the Table over a set of columns.
           P(X, Y, Z) -> P(X, Y) or P(X, Z) or P(Y, Z)
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
        # check the validity of operation based on column names
        if len(args) == self.columns.size:
            raise ValueError("Cannot marginalize on all column names.")

        # split columns to indices and comp_indices
        columns_info = self.columns.split_columns(*args)
        #
        # Convert the key:values to 2D numpy array
        # the array rows are (row, value)
        arr = self.to_2d_array()
        # filter the compliment columns
        filtered_arr = np.c_[arr[:, columns_info.complimnet_indices], arr[:, -1]]
        # split the 2d array's rows to a tuple of
        # compliment columns (row[comp_indices])
        # and count row[-1]
        arr_gen = ((RowKey(row[:-1]), row[-1]) for row in filtered_arr)
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
        table = Table(grouped_arr, columns_info.complimnet_names, _internal_=True)
        if normalise:
            table.normalise()

        return table

    def condition_on(self, *args, normalise=True):
        """Creates the conditional based on
           the provided names of columns.
           P(X, Y) -> P(X | Y) or P(Y | X)
        Args:
            args (list):
                List of names of provided random
                variables.

        Raises:
            ValueError:
                If the provided RV names do not exist
                in the distribution.

        Returns:
            MultiTable
        """
        if self.columns.size == 1:
            raise ValueError("This is a single column Table and cannot condition on.")

        if len(args) == self.columns.size:
            raise ValueError("Cannot condition on all columns.")
        # split columns to indices and comp_indices
        columns_info = self.columns.split_columns(*args)
        # Convert the key:value to 2D numpy array
        # the array rows are (rows, value)
        arr = self.to_2d_array()
        # divide the 2d array's rows to a tuple of columns,
        # (row[indices]), compliment columns (row[comp_indices])
        # and values row[-1]
        arr_gen = (
            (
                RowKey(row[columns_info.indices]),
                RowKey(row[columns_info.complimnet_indices]),
                row[-1],
            )
            for row in arr
        )
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
        table = MultiTable(
            {
                key: Table(values, columns_info.complimnet_names, _internal_=True)
                for key, values in grouped_arr.items()
            },
            columns_info.indices_names,
        )
        if normalise:
            table.normalise()

        return table

    def reduce(self, **kwargs):
        """Reduce the Table by one or more columns.
           P(X, Y) -> P(X = x, Y) or  P(X,  Y = y)
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
        # split columns to indices and comp_indices
        columns = list(kwargs.keys())
        if len(columns) == self.columns.size:
            raise ValueError("Cannot reduce on all column names.")
        columns_info = self.columns.split_columns(*columns)
        values = np.array([value for _, value in kwargs.items()], dtype=np.object)
        #
        # Convert the key:values to 2D numpy array
        # the array rows are (keys, value)
        arr_counter = self.to_2d_array()
        # filter the 2d array rows by provided values of the reduce
        # conditioned_arr is a boolean one, and filtering happens
        # in the second line
        conditioned_arr = np.all(arr_counter[:, columns_info.indices] == values, axis=1)
        sliced_arr = arr_counter[conditioned_arr, :]
        # filter the 2d array columns (the compliment columns)
        # plus the value column (which is the last column)
        sliced_arr = sliced_arr[:, columns_info.complimnet_indices + [-1]]
        # divide the 2d array's rows to a tuple of columns
        # and value
        # So, we make a generator that divide the rows to the tuple of
        # columns (tuple(row[:-1]) and value (row[-1])
        arr_gen = ((RowKey(row[:-1]), row[-1]) for row in sliced_arr)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of column (index zero in itemgetter)
        sorted_slice_arr = sorted(arr_gen, key=itemgetter(0))
        # group by the filtered columns (compliment
        # columns) and sum the value per key
        # Note that the 'itemgetter' read the first index which
        # is the tuple of compliment columns
        return Table(
            {
                k: sum([item[1] for item in g])
                for k, g in groupby(sorted_slice_arr, key=itemgetter(0))
            },
            columns_info.complimnet_names,
            _internal_=True,
        )

    def get(self, *args, **kwargs):
        key = self.columns.to_key(*args, **kwargs)
        return super().__getitem__(key)

    def to_table(self, sort=False, value_title=""):

        arr = self.to_2d_array().astype("U")
        arr_len = np.apply_along_axis(lambda row: [len(item) for item in row], 0, arr)
        max_levels_len = np.max(arr_len[:, :-1], axis=0)

        max_freq_len = max(np.max(arr_len[:, -1]), len(value_title))

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

    def add(self, that):
        """Combines two FrequencyTable and return
        a new one. All the frequencies are sum together.
        This is not a mathematical sum.
        """
        #############################################
        # check the validity of operation based on column names
        if not isinstance(that, Table):
            raise ValueError("Table can only adds to Table.")

        if self.columns.size != that.columns.size:
            raise ValueError("Two adding Table do not have the same columns.")

        if len(self.children_names) != len(that.children_names):
            raise ValueError("Two adding Table do not have the same children columns.")

        for i, name in enumerate(self.names):
            if name != that.names[i]:
                raise ValueError(
                    "Two adding Table do not have the same columns "
                    "(order must be the same too)."
                )

        for i, name in enumerate(self.children_names):
            if name != that.children_names[i]:
                raise ValueError(
                    "Two adding Table do not have the same children columns "
                    "(order must be the same too)."
                )
        #############################################
        #  Algorithm
        #

        def add_internal(this, that, names):
            if that is not None:
                for key in that.keys():
                    if key in this:
                        this[key] += that[key]
                    else:
                        this[key] = that[key]

            return Table(this, names=names, _internal_=True)

        ############################################
        # MultiTable handeling
        if self.columns.is_multitable():
            return Table(
                {
                    k: add_internal(table.copy(), that[k], self.children_names)
                    for k, table in self.items()
                },
                self.names,
                _internal_=True,
            )

        return add_internal(self.copy(), that, self.names)

    def total(self):
        if self.columns.is_multitable():
            return {k: table.total() for k, table in self.items()}

        return sum(self.values())

    def normalise(self):
        if self.columns.is_multitable():
            for k, total in self.total().items():
                if total == 0:
                    continue
                table = self[k]
                for k2 in table:
                    table[k2] /= total

        else:
            total = self.total()
            if total != 0:
                for k in self.keys():
                    self[k] /= total

    def __mul__(self, right):
        """Multiplies a table with this one.
           P(X, Y) * k -> P(X, Y)
           P(X) * P(Y, Z) -> P(X, Y, Z)
        Args:
            right ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        if not isinstance(right, Table):
            raise ValueError("The 'right' argument must be a 'Table'.")

        (rows, names) = self._product_(right)
        return Table(rows, names, _internal_=True)

    def __rmul__(self, left):
        """Multiplies a table with this one.
           k * P(X, Y) -> P(X, Y)
           P(X) * P(Y, Z) -> P(X, Y, Z)
        Args:
            right ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if not isinstance(left, Table):
            raise ValueError("The 'right' argument must be a 'Table'.")

        (rows, names) = left._product_(self)
        return Table(rows, names, _internal_=True)

    def __add__(self, right):
        return self.add(right)


def prod_right(table, key2, value2):
    # Product a table with kay and value
    if value2 is None:
        return {}
    return {key1 + key2: value1 * value2 for key1, value1 in table.items()}


def prod_left(table, key2, value2):
    # Product a table with kay and value
    if value2 is None:
        return {}
    return {key2 + key1: value1 * value2 for key1, value1 in table.items()}


def multi_table_to_table_product(left, right, all_ordered_names):
    """Multiply two tables.
    P(X, Y | Z) * P(Z) -> P(X, Y, Z)
    P(X, Y | Z, W) * P(Z) -> P(X, Y, Z | W)
    """
    # Case P(X, Y | Z) * P(Z) -> P(X, Y, Z)
    if list(left.names) == list(right.names):
        return Table(
            ChainMap(
                *[
                    prod_right(table, key2=k, value2=right[k])
                    for k, table in left.items()
                ]
            ),
            left.columns.children_names + left.names,
            _internal_=True,
        )
    # Case P(X, Y | Z, W) * P(Z) -> P(X, Y, Z | W)
    for name in right.names:
        if not left.columns:
            raise ValueError(
                f"Column name '{name}'in right table is not defined on "
                "conditioned columns of the left Table (name mismatched)."
            )
    # e.g. P(X, Y | Z, W) * P(Z) : indices of [W]
    indices = [i for i, name in enumerate(left.names) if name not in right.names]
    # e.g. P(X, Y | Z, W) * P(Z) : indices of [Z]
    compliment_indices = [i for i in range(left.columns.size) if i not in indices]
    # e.g. P(X, Y | Z, W) * P(Z) : [W]
    reduced_names = [left.names[i] for i in indices]
    children_names = [
        names for names in all_ordered_names if names not in reduced_names
    ]

    def reduced_key(key):
        # Method to split the keys
        return {left.names[i]: key[i] for i in indices}

    def compliment_key(key):
        # Method to make a split key
        return RowKey(*[key[i] for i in compliment_indices])

    # Case: P(X, Y | Z, W) * P(Z) -> P(X, Y, Z | W)
    if right.columns.size == len(indices):
        return MultiTable(
            ChainMap(
                *[
                    prod_right(table, key2=k, value2=right[k])
                    for k, table in left.items()
                ]
            ),
            reduced_names,
            _children_names_=children_names,
        )

    return MultiTable(
        {
            compliment_key(k): table * right.reduce(**reduced_key(k))
            for k, table in left.items()
        },
        reduced_names,
        _children_names_=children_names,
    )


def table_to_multi_table_product(left, right, all_ordered_names):
    """Multiply two tables.
    P(Z) * P(X, Y | Z)  -> P(Z, X, Y)
    P(Z) * P(X, Y | Z, W) -> P(Z, X, Y | W)
    """
    # Case P(Z) * P(X, Y | Z) -> P(Z, X, Y)
    if list(left.names) == list(right.names):
        return Table(
            ChainMap(
                *[
                    prod_left(table, key2=k, value2=left[k])
                    for k, table in right.items()
                ]
            ),
            right.names + right.columns.children_names,
            _internal_=True,
        )
    # Case P(Z) * P(X, Y | Z, W) -> P(Z, X, Y | W)
    for name in left.names:
        if not right.columns:
            raise ValueError(
                f"Column name '{name}'in left table is not defined on "
                "conditioned columns of the right Table (name mismatched)."
            )
    # e.g. P(Z) * P(X, Y | Z, W) : indices of [W]
    indices = [i for i, name in enumerate(right.names) if name not in left.names]
    # e.g. P(Z) * P(X, Y | Z, W) : indices of [Z]
    compliment_indices = [i for i in range(right.columns.size) if i not in indices]
    # e.g. P(Z) * P(X, Y | Z, W) : [W]
    reduced_names = [right.names[i] for i in indices]
    children_names = [
        names for names in all_ordered_names if names not in reduced_names
    ]

    def reduced_key(key):
        # Method to split the keys
        return {right.names[i]: key[i] for i in indices}

    def compliment_key(key):
        # Method to make a split key
        return RowKey(*[key[i] for i in compliment_indices])

    # Case: P(Z) * P(X, Y | Z, W) -> P(Z, X, Y | W)
    if left.columns.size == len(indices):
        return MultiTable(
            ChainMap(
                *[
                    prod_left(table, key2=k, value2=left[k])
                    for k, table in right.items()
                ]
            ),
            reduced_names,
            _children_names_=children_names,
        )

    return MultiTable(
        {
            compliment_key(k): table * left.reduce(**reduced_key(k))
            for k, table in right.items()
        },
        reduced_names,
        _children_names_=children_names,
    )


def multi_table_to_multi_table_product(table_main, table_side, all_ordered_names):
    indices = [
        i for i, name in enumerate(table_main.names) if name not in table_side.names
    ]
    compliment_indices = [i for i in range(table_main.columns.size) if i not in indices]
    reduced_names = [table_main.names[i] for i in compliment_indices]
    children_names = [
        names for names in all_ordered_names if names not in reduced_names
    ]

    def reduced_key(key):
        # Method to split the keys
        return {table_main.names[i]: key[i] for i in indices}

    def compliment_key(key):
        # Method to split the keys
        return RowKey(*[key[i] for i in compliment_indices])

    if len(table_side.columns.children_names) == len(indices):

        def prod2(key1, table1):
            table_side_table2 = table_side[key1]
            if table_side_table2 is None:
                return {}
            return {
                compliment_key(key1): table1 * table2
                for key2, table2 in table_side_table2
            }

        return MultiTable(
            ChainMap(*[prod2(key1, table1) for key1, table1 in table_main.items()]),
            reduced_names,
            _children_names_=children_names,
        )

    return MultiTable(
        {
            compliment_key(key1): table1 * table2
            for key1, table1 in table_main.items()
            for key2, table2 in table_side.reduce(**reduced_key(key1))
        },
        reduced_names,
        _children_names_=children_names,
    )


def multi_table_product(left, right):
    """Multiply two tables.
        P(X, Y | Z) * P(Z) -> P(X, Y , Z)
        P(X, Y | Z, W) * P(Z) -> P(X, Y , Z | W)
        P(X, Y | Z, U) * P(Z | U) -> P(X, Y , Z | U)
        P(X, Y | Z, U, W) * P(Z | U, W) -> P(X, Y , Z | U, W)

        in the case of two conditionals, the longer one defines
        the order of variables
        e.g.
        P(X, Y | Z, U, W) * P(Z | W, U) -> P(X, Y , Z | U, W)
        P(Z | W, U) * P(X, Y | Z, U, W) -> P(X, Y , Z | U, W)


    Args:
        left ([type]): [description]
        right ([type]): [description]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # Cases:
    # P(X, Y | Z) * P(Z) -> P(X, Y, Z)
    # P(X, Y | Z, W) * P(Z) -> P(X, Y, Z | W)
    if not isinstance(right, MultiTable):
        if sorted(right.names) != sorted(left.names):
            raise ValueError("The right names is" " not equal to conditionals of left.")

        all_ordered_names = left.columns.children_names + right.columns.names
        return multi_table_to_table_product(left, right, all_ordered_names)

    # Cases:
    # P(Z) * P(X, Y | Z) -> P(Z, X, Y)
    # P(Z) * P(X, Y | Z, W) -> P(Z, X, Y | W)
    if not isinstance(left, MultiTable):
        if sorted(right.names) != sorted(left.names):
            raise ValueError("The left names is" " not equal to conditionals of right.")

        all_ordered_names = left.names + right.columns.children_names
        return table_to_multi_table_product(left, right, all_ordered_names)

    # Cases:
    # P(X, Y | Z, U) * P(Z | U) -> P(X, Y, Z | U)
    # P(X, Y | Z, U, W) * P(Z | U, W) -> P(X, Y, Z | U, W)
    # P(X, Y, Z|  U, W) * P(U | W) -> P(X, Y, Z, U | W
    # P(X, Y, Z|  U, V, W) * P(U, V | W) -> P(X, Y, Z, U, V | W)
    def in_the_other(first, second):
        for name in first:
            if name not in second:
                return False
        return True

    common_conditions = [name for name in left.names if name in right.names]
    right_compliment_conditions = [
        name for name in right.names if name not in common_conditions
    ]
    left_compliment_conditions = [
        name for name in left.names if name not in common_conditions
    ]

    # To check the crossed cases
    # e.g. P(X | Y) * P(Y | X)
    # after removing common names on conditionals,
    # one of them must remains conditionless
    # e.g.
    # 1) P(X, Y | Z, U) * P(Z | U)
    #    removes commons: P(X, Y | Z) * P(Z)
    # 2) P(Z | U, W) * P(X, Y | Z, U, W)
    #    removes commons: P(Z) * P(X, Y | Z)
    # 3) P(X | Y) * P(Y | X)
    #    remove commons fails
    if len(right_compliment_conditions) == 0:
        if not in_the_other(right.columns.children_names, left.names):
            raise ValueError(
                "Columns in right is not defined in conditional names of left."
            )
        all_ordered_names = left.columns.children_names + right.columns.children_names
        return multi_table_to_multi_table_product(left, right, all_ordered_names)

    elif len(left_compliment_conditions) == 0:
        if not in_the_other(left.columns.children_names, right.names):
            raise ValueError(
                "Columns in left is not defined in conditional names of right."
            )
        all_ordered_names = left.columns.children_names + right.columns.children_names
        return multi_table_to_multi_table_product(right, left, all_ordered_names)
    else:
        raise ValueError("Columns and conditional names mismatch.")


class MultiTable(Table):
    def __init__(self, rows, names=None, _children_names_=None):
        super().__init__(
            rows, names, _internal_=True, _children_names_=_children_names_
        )

    def marginal(self, *args, normalise=True):
        """[summary]
           P(X, Y | Z) -> P(X | Z) or P(Y | Z)
        Args:
            normalise (bool, optional): [description]. Defaults to True.

        Raises:
            ValueError: [description]

        Returns:
            MultiTable: [description]
        """

        for name in args:
            if name in self.names:
                raise ValueError(f"Cannot marginalize on conditioned columns:'{name}'.")

        table = Table(
            {
                k: table.marginal(*args, normalise=normalise)
                for k, table in self.items()
            },
            self.names,
            _internal_=True,
        )

        if normalise:
            table.normalise()

        return table

    def condition_on(self, *args, normalise=True):
        """Creates the conditional based on
           the provided names of columns.

           P(X, Y | Z) -> P(X | Y, Z) or P(Y | X, Z)

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
        for name in args:
            if name in self.names:
                raise ValueError(f"Cannot condition on conditioned columns:'{name}'.")
        conditioned_children = (
            (k, table.condition_on(*args, normalise=normalise))
            for k, table in self.items()
        )

        return MultiTable(
            {
                key2 + key1: table
                for key1, key2_table in conditioned_children
                for key2, table in key2_table.items()
            },
            # It results in: P(X, Y | Z) -> P(X | Y, Z)
            # inversing the order turns it P(X, Y | Z) -> P(X | Z, Y)
            # Maybe more controls is needed here
            list(args) + self.names,
        )

    def reduce(self, **kwargs):
        """Reduce the Table by one or more columns.
            P(X, Y | Z) -> P(X = x, Y | Z) or  P(X,  Y = y | Z)
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
        return MultiTable(
            {k: table.reduce(**kwargs) for k, table in self.items()},
            self.names,
        )

    def __mul__(self, right):
        if not isinstance(right, Table):
            raise ValueError("The 'right' argument must be a 'Table'.")

        return multi_table_product(self, right)

    def __rmul__(self, left):
        if not isinstance(left, Table):
            raise ValueError("The 'left' argument must be a 'Table'.")

        return multi_table_product(left, self)
