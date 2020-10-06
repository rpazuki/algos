from numbers import Number
from abc import ABC, abstractmethod
import numpy as np


class Key(tuple):
    def __new__(cls, value):
        try:
            if isinstance(value, str) or isinstance(value, Number):
                return tuple.__new__(cls, (value,))
            return tuple.__new__(cls, value)
        except TypeError:  # e.g. key is int
            return tuple.__new__(cls, (value,))


class RandomVariable(ABC):
    def __init__(self, size):
        self.size = size

    @abstractmethod
    def to_key(self, *args, **kwargs):
        pass

    @abstractmethod
    def to_dict_key(self, *args, **kwargs):
        pass


class DiscreteRV(RandomVariable):
    """Store the details of single discrete random variable."""

    def __init__(self, name, first_row):
        """Store the details of single discrete random variable.

        Args:
            name (str):
                The discrete random variable's name.
            first_row (object):
                An example of independent variable to extract the
                information about random variable.
        """
        super().__init__(size=1)
        self.name = str(name)
        self.is_numeric = isinstance(first_row, Number)

    def to_key(self, *args, **kwargs):
        total_size = len(args) + len(kwargs.keys())
        if total_size != 1:
            raise ValueError(
                f"Random variable '{self.name}' can accept one level,"
                f" {total_size} provided."
            )
        if len(args) == 1:
            return args[0]

        key_name = next(iter(kwargs.keys()))
        if key_name != self.name:
            raise ValueError(
                f"The provided name '{key_name}' is not"
                f" defined for random variable '{self.name}'"
            )
        return kwargs[key_name]

    def to_dict_key(self, *args, **kwargs):
        return {self.name: self.to_key(*args, **kwargs)}

    def __len__(self):
        return 1

    def __str__(self):
        return f"'{self.name}'"

    __repr__ = __str__


class MultiDiscreteRV(RandomVariable):
    """Store the details of or more discrete random variables."""

    def __init__(self, first_row, names=None, variable_name="X"):
        """Store the details of or more discrete random variables.

           it can generate the names of random variables if it is Note
           provided.

        Args:
            first_row (object or a tuple):
                An example of independent variables to extract the
                information about random variable(s).
            names (list, optional):
                A list of random variable names. Defaults to None.
            variable_name (str, optional):
                The prefix for automatic name generation.
                Defaults to "X".

        Raises:
            ValueError: When the length of provided names is not equal to
                        the length of tuples in 'factors'.
            ValueError: When the length of tuples in 'factors' are not
                        equal.
        """
        first_row_key = Key(first_row)
        rv_len = len(first_row_key)
        # Check names, if it is None, create one equal to length
        # of  random variables
        if names is None:
            self.names = np.array([f"{variable_name}{i+1}" for i in range(rv_len)])
        elif len(names) != rv_len:
            raise ValueError(
                f"'factors' has {rv_len} random variables while"
                f"'names' argument has {len(names)}."
            )
        else:
            self.names = np.array(names)
        # Store the DiscreteRV as a dictionary
        if rv_len > 1:
            self.multi_rvs = {
                name: DiscreteRV(name, item)
                for name, item in zip(self.names, first_row)
            }
        else:
            self.multi_rvs = {self.names[0]: DiscreteRV(self.names[0], first_row)}
        # The size of the MultiDiscreteRV is the same
        # as the number RVs or their names
        size = len(self.names)
        super().__init__(size=size)
        #

    def to_key(self, *args, **kwargs):
        total_size = len(args) + len(kwargs.keys())
        if total_size != self.size:
            raise ValueError(
                f"Multi-random variables '{self.names}' can accept {self.size} "
                f"number of levels, {total_size} provided."
            )

        if len(args) == self.size and self.size > 1:
            return tuple(args)
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
            return tuple(return_list)

    def to_dict_key(self, *args, **kwargs):
        return {
            key: value for key, value in zip(self.names, self.to_key(*args, **kwargs))
        }

    def __getitem__(self, rv_index):
        """An indexer by position (int) or name (str).

        Args:
            rv_index (int or str): Random variable's name or index.

        Returns:
            DiscreteRV: An instance of DiscreteRV.
        """
        if isinstance(rv_index, int):
            name = self.names[rv_index]
            return self.multi_rvs[name]
        elif isinstance(rv_index, str):
            return self.multi_rvs[rv_index]
        else:
            raise ValueError("The provided index is not 'int' or 'str'.")

    def index_of(self, name):
        """Finds the index of the random variable from its name.

        Args:
            name (str): Name of the random variable.

        Returns:
            int: Index of the random variable.
        """
        indices = [i for i, n in enumerate(self.names) if n == name]
        if len(indices) == 0:
            return -1
        else:
            return indices[0]

    def __len__(self):
        return len(self.names)

    def __str__(self):
        return "".join([f"{s}\n" for s in self.multi_rvs.values()])

    __repr__ = __str__

    def __contains__(self, name):
        """Check the name of the random variable.

        Args:
            name (str): Name of the random variables.

        Returns:
            [bool]: True or False
        """
        return name in self.multi_rvs


class Distribution(ABC):
    def prob(self, *args, **kwargs):
        key = self._get_random_variable_().to_key(*args, **kwargs)
        return self.probability(key)

    def __iter__(self):
        return iter(self.keys())

    def levels(self):
        arr = self._to_2d_array_()
        # the last column is count, so we drop it
        if len(arr.shape) < 2:  # empty dist
            return np.array([])

        column_num = arr.shape[1] - 1
        arr = arr[:, :-1]
        if column_num == 1:
            return np.unique(arr)

        return np.array([np.unique(arr[:, i]) for i in range(column_num)])

    def _to_2d_array_(self):
        """Convert the distribution ( or the self._counter's
           key:value) to a 2D numpy array where the array
           rows are [[(RV_1, RV_2, ..., RV_n, count)],[...

        Returns:
            numpy ndarray:
                A 2D numpy array that the its last column
                is the counts.
        """
        if self._get_random_variable_().size == 1:
            return np.array([(k, v) for k, v in self.items()], dtype=np.object)

        return np.array([tuple(k) + (v,) for k, v in self.items()], dtype=np.object)

    @abstractmethod
    def normalise(self):
        pass

    @abstractmethod
    def probability(self, key):
        pass

    @abstractmethod
    def keys(self):
        pass

    @abstractmethod
    def items(self):
        pass

    @abstractmethod
    def _get_random_variable_(self):
        pass

    @abstractmethod
    def __mul__(self, that):
        pass

    @abstractmethod
    def __rmul__(self, that):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    @abstractmethod
    def __contains__(self, key):
        pass
