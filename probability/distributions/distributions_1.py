from numbers import Number
from abc import ABC, abstractmethod
from collections import Counter
import numpy as np


class Key:
    def __init__(self, value):
        try:
            if isinstance(value, str):
                self.value = (value,)
            else:
                self.value = tuple(value)
            self.is_tuple = True
        except TypeError:  # e.g. key is int
            self.value = value
            self.is_tuple = False

    def __getitem__(self, key):
        if self.is_tuple:
            return self.value[key]
        else:
            return self.value

    def __mul__(self, right):
        if self.is_tuple and right.is_tuple:
            return self.value + right.value

        if self.is_tuple:
            return self.value + (right.value,)

        if right.is_tuple:
            return (self.value,) + right.value

        return (self.value, right.value)


class RandomVariable(ABC):
    def __init__(self, levels, size):
        self.levels = levels
        self.size = size

    @abstractmethod
    def to_key(self, *args, **kwargs):
        pass

    @abstractmethod
    def to_dict_key(self, *args, **kwargs):
        pass


class DiscreteRV(RandomVariable):
    """Store the details of single discrete random variable."""

    def __init__(self, name, levels):
        """Store the details of single discrete random variable.

        Args:
            name (str): The discrete random variable's name.
            levels (set): discrete levels.
        """
        super().__init__(set(levels), size=1)
        self.name = str(name)
        self.is_numeric = all([isinstance(level, Number) for level in self.levels])

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
        return f"'{self.name}': {self.levels}"

    def __repr__(self):
        return self.__str__()


class MultiDiscreteRV(RandomVariable):
    """Store the details of or more discrete random variables."""

    def __init__(self, factors, names=None, variable_name="X"):
        """Store the details of or more discrete random variables.

           it can generate the names of random variables if it is Note
           provided.

        Args:
            factors (iter): A list of objects or tuples that contains the independent
                            variables' values.
            names (list, optional): A list of random variable names. Defaults to None.
            variable_name (str, optional): The prefix for automatic name generation.
                                           Defaults to "X".

        Raises:
            ValueError: When the length of provided names is not equal to
                        the length of tuples in 'factors'.
            ValueError: When the length of tuples in 'factors' are not
                        equal.
        """
        # To check the consistency of the tuples, save
        # the length of the first one and check all the others
        first_row = next(iter(factors))
        if isinstance(first_row, tuple):
            rv_len = len(first_row)
        else:
            rv_len = 1
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
        # Each RV has its own set of levels
        if rv_len > 1:
            first_row_t = tuple(first_row)
            levels = np.array([{first_row_t[i]} for i in range(rv_len)])
        else:
            levels = np.array([{first_row}])
        # We suppose the classes were tuples
        # and each Random Variable (RV) is positioned
        # in a fix place of the n-tuple.
        # Therefore, the levels of the RV can be
        # found by iterating over each tuple's item
        if rv_len > 1:
            # Convert each features line to tuple
            tuples = (tuple(row) for row in factors)
            for row in tuples:
                if len(row) != rv_len:
                    raise ValueError("The length of the 'factors' is not consistence.")
                for i, level in enumerate(row):
                    levels[i] |= {level}
        else:
            levels[0] = {first_row} | set(factors)
        # Store the DiscreteRV as a dictionary
        self.multi_rvs = {
            name: DiscreteRV(name, l) for name, l in zip(self.names, levels)
        }
        # The size of the MultiDiscreteRV is the same
        # as the number RVs or their names
        size = len(self.names)
        np_levels = np.array([item for item in levels])
        super().__init__(levels=np_levels, size=size)
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

    def __repr__(self):
        return self.__str__()

    def __contains__(self, name):
        """Check the name of the random variable.

        Args:
            name (str): Name of the random variables.

        Returns:
            [bool]: True or False
        """
        return name in self.multi_rvs


class Distribution(ABC):
    def __init__(self, samples):
        """Construct an abstract distribution and count the number of
        occurenc of items in the samples.

        Args:
            samples (iterable):
                An iterable that contains the observed samples
                or a dictionary of (key:frequency).

        Raises:
            ValueError: Raises when the provided sample is None.
        """
        if samples is None:
            raise ValueError("samples argument is None.")

        self._counter = Counter(samples)
        # Elements count
        self.total = sum(self._counter.values())

    @staticmethod
    def digitize(samples, start, stop, num=10, endpoint=True, right=False, levels=None):
        """[summary]

        Args:
            samples (numeric array):
                continouse values that needs digitization.
            start (numeric):
                The starting value of the sequence.
            stop (numeric):
                The end value of the sequence, unless `endpoint` is set to False.
                In that case, the sequence consists of all but the last of ``num + 1``
                evenly spaced samples, so that `stop` is excluded.  Note that the step
                size changes when `endpoint` is False.
            num (int, optional):
                Number of samples to generate. Default is 10. Must be non-negative.
            endpoint (bool, optional):
                If True, `stop` is the last sample. Otherwise, it is not included.
                Defaults to True.
            right (bool):
                Indicating whether the intervals include the right or the left bin
                edge. Default behavior is (right==False) indicating that the interval
                does not include the right edge. The left bin end is open in this
                case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
                monotonically increasing bins.
            levels (list, optional):
                List of labels for each step. Defaults to None.

        Returns:
            numpy array:
                [type]: An array of bins/levels of digitized samples.
        """
        bins = np.linspace(start, stop, num, endpoint=endpoint)
        return Distribution.digitize_bin(samples, bins, right, levels)

    @staticmethod
    def digitize_bin(samples, bins, right=False, levels=None):
        """Return the digitized samples by returning the coresponding bins value.
           When 'levels' is provided, bins values replace by levels.

           =========  =============  ============================
           `right`    order of bins  returned index `i` satisfies
           =========  =============  ============================
           ``False``    increasing   ``bins[i-1] <= x < bins[i]``
           ``True``     increasing   ``bins[i-1] < x <= bins[i]``
           ``False``    decreasing   ``bins[i-1] > x >= bins[i]``
           ``True``     decreasing   ``bins[i-1] >= x > bins[i]``
           =========  =============  ============================

        Args:
            samples (numeric array):
                continouse values that needs digitization.
            bins (array):
                Array of bins. It has to be 1-dimensional and monotonic.
            right (bool):
                Indicating whether the intervals include the right or
                the left bin edge. Default behavior is (right==False)
                indicating that the interval does not include the right
                edge. The left bin end is open in this case, i.e.,
                bins[i-1] <= x < bins[i] is the default behavior for
                monotonically increasing bins.
            levels (list, optional):
                List of labels for each step. Defaults to None.

        Raises:
            ValueError:
                Raises when the length of levels is not equal to
                the length of bins minus one.

        Returns:
            numpy array:
                An array of bins/levels of digitized samples.
        """
        if levels is not None and len(levels) != len(bins) - 1:
            raise ValueError(
                f"'levels' length ({len(levels)}) is not "
                f"equal to bins length-1 ({len(bins)-1})."
            )
        indices = np.digitize(samples, bins, right)

        if levels is None:
            # Extend the bins to include left outliers
            delta_left = bins[1] - bins[0]
            bins_extended = np.r_[[bins[0] - delta_left], bins]
            return bins_extended[indices]

        # Extend the levels to include outliers
        levels_extended = np.r_[["less"], levels, ["more"]]
        return levels_extended[indices]

    def normalise(self):
        """Normalise the distribution."""
        for k in self._counter:
            self._counter[k] = self._counter[k] / self.total
        self.total = 1.0

    def prob(self, *args, **kwargs):
        key = self._get_random_variable_().to_key(*args, **kwargs)
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
        if self.total == 0:
            return 0

        return self.__getitem__(key) / self.total

    def freq(self, *args, **kwargs):
        key = self._get_random_variable_().to_key(*args, **kwargs)
        return self.frequency(key, normalised=False)

    def frequency(self, key, normalised=False):
        """Gets the frequency of the random variable, when its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object):
                the value of the random variable.
            normalised (bool, optional):
                normalize the return. Defaults to False.

        Returns:
            int or float: frequency or probability of the random variable.
        """
        if self.total == 0:
            return 0

        if normalised:
            return self._counter[key] / self.total
        else:
            return self._counter[key]

    def keys(self):
        return self._counter.keys()

    def items(self):
        return self._counter.items()

    def frequencies(self, normalised=True):
        """A list of frequencies of class occurenc.

        Args:
            normalised (bool, optional):
                The normalisation flag. Defaults to True.

        Returns:
            list: A list of floats or integers, depending of normalisation.
        """
        if self.total == 0:
            return np.zeros(len(self._counter.keys()))

        values = np.array(list(self._counter.values()))

        if normalised:
            return np.array(values) / self.total

        return np.array(values)

    def keys_as_list(self):
        return [k for k in self._counter.keys()]

    def most_common(self, num: int = None):
        """List the n most common elements and their counts from the most
            common to the least. If n is None, then list all element counts.

        Args:
            num (int, optional):
                The maximum length of the returned list. Defaults to None.

        Returns:
            list: A list of tuples. The first element of the tuple is a class
                  key and th second one is its count.
        """
        return self._counter.most_common(num)

    @abstractmethod
    def _get_random_variable_(self):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def to_table(self, normalised=False, sort=False):
        pass

    @abstractmethod
    def __mul__(self, that):
        pass

    @abstractmethod
    def __rmul__(self, that):
        pass

    def __getitem__(self, key):
        """An indexer that returns the count of the class key.

            Returns zero if the 'key' does not exist in samples or
            the samples iterator is empty.


        Args:
            key (object):
                The key that specifies the class name in samples.

        Returns:
            [float]: The count of occurrence of class 'key'.
                     Returns zero if the 'key' does not exist in samples or
                     the samples iterator is empty.
        """
        if self.total == 0:
            return 0

        if isinstance(key, slice):
            return list(self._counter.items())[key]

        return self._counter[key]

    def __contains__(self, key):
        return key in self._counter

    def __iter__(self):
        return iter(self._counter.keys())
