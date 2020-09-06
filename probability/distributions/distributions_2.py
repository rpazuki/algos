import numpy as np
from probability.distributions import Key
from probability.distributions import DiscreteRV
from probability.distributions import MultiDiscreteRV
from probability.distributions import Distribution

from probability.distributions.distributions_3 import ConditionalDistribution


class FrequencyTable(Distribution):
    """Provides a frequency table from the number of occurenc of
    observed items as dictionary of (key:frequency) or an iterator
    of observed sample.
    """

    def __init__(self, samples, name="X1"):
        """Construct a FrequencyTable from the number of occurenc in samples.

        Args:
            samples (iterable): An iterable that contains the observed samples
                                or a dictionary of (key:frequency).
            name (str): Name of the random variable.

        Raises:
            ValueError: Raises when the provided sample is None.
        """
        super().__init__(samples)
        # Random varable's details
        self.discrete_rv = DiscreteRV(name, levels=self._counter.keys())
        self.name = name

    @classmethod
    def from_txt(cls, fname, col, dtype="U", name=None, converter=None, **kwargs):
        """Calls numpy genfromtxt to load the data.

        Args:
            fname (file, str, pathlib.Path): file name
            col (int): column number
            dtype (dtype, optional): Data type of the resulting array.
                   If None, the dtype will be determined by the contents
                   of each column, individually. Defaults to "U" to convert
                   to unicode text.
            name (str, optional): name of the
                                  random variable.
                                  If it is not provided, it creates as 'X1'.
                                  Defaults to None.
            converter (function): calls to convert the columns or to handel
                                  missing values. Defaults to None.
        """
        if "names" in kwargs:
            raise ValueError(
                "'names' argument cannot be used for this function. "
                "It can only load a single column. Use 'name' instead."
            )
        if "usecols" in kwargs:
            raise ValueError(
                "'usecols' argument cannot be used for this function. "
                "It can only load a single column. Use 'col' instead."
            )
        if "dtypes" in kwargs:
            raise ValueError(
                "'dtypes' argument cannot be used for this function. "
                "It can only load a single column. Use 'dtype' instead."
            )
        if "converters" in kwargs:
            raise ValueError(
                "'converters' argument cannot be used for this function. "
                "It can only load a single column. Use 'converter' instead."
            )

        if col is not None:
            kwargs["usecols"] = [col]
        else:
            raise ValueError("'col' argument is None.")

        if dtype is not None:
            dtypes = [dtype]
        else:
            dtypes = None

        if converter is not None:
            kwargs["converters"] = {col: converter}

        samples = np.genfromtxt(fname, dtypes, **kwargs)
        return cls(samples=samples, name=name)

    def product(self, right):
        """Multiplies a Distribution to this one.

        Args:
            right ([Distribution]): the other Distribution.

        Raises:
            ValueError: Raises When the right argument is not a
                        subclass of Distribution.

        Returns:
            [type]: [description]
        """
        if not isinstance(right, Distribution):
            raise ValueError("The 'right' argument must be a Distribution.")
        # Make sure the right is FrequencyTable table too
        # otherwise, ask the other type to handel the multiplication
        if not isinstance(right, FrequencyTable):
            return right.__rmul__(self)
        # Here , we can safely assume both sides are FrequencyTable
        # and Multiply them
        if right is self:
            names = [f"{self.discrete_rv.name}1", f"{self.discrete_rv.name}2"]
        elif self.discrete_rv.name == right.discrete_rv.name:
            names = [f"{self.discrete_rv.name}1", f"{self.discrete_rv.name}2"]
        else:
            names = [self.discrete_rv.name, right.discrete_rv.name]
        # The multiplication of two FrequencyTable must
        # be a DiscreteDistribution
        return DiscreteDistribution(
            {
                Key(k1) * Key(k2): v1 * v2
                for k1, v1 in self.items()
                for k2, v2 in right.items()
            },
            names,
        )

    def _get_random_variable_(self):
        return self.discrete_rv

    def summary(self):
        return (
            "Frequency table \n"
            f"random variable:'{self.discrete_rv.name}'\n"
            f"levels: {self.discrete_rv.levels}\n"
            f"total:{self.total}"
        )

    def to_table(self, normalised=False):

        title = "probability" if normalised else "frequency"
        total_title = "**total**"

        norm = self.total if (normalised and self.total != 0) else 1

        max_level_len = np.max(
            [len(str(level)) for level in self.discrete_rv.levels]
            + [len(self.name), len(total_title)]
        )

        max_freq_len = np.max(
            [len(str(value / norm)) for _, value in self.items()]
            + [len(str(self.total)), len(title)]
        )

        def padding(max_len):
            def str_padding(value):
                return "".join([" "] * (max_len - len(str(value))))

            return str_padding

        l_padding = padding(max_level_len)
        r_padding = padding(max_freq_len)

        rows = ""
        for k, value in sorted(self.items()):
            padding = l_padding(k)
            freq_padding = r_padding(value / norm)
            rows += f"|{padding}{k}|{value/norm}{freq_padding}|\n"

        name_padding = l_padding(self.name)
        value_padding = r_padding(title)
        total_title_padding = l_padding(total_title)
        total_padding = r_padding(self.total)

        horizontal_line = "".join(
            ["|"] + ["-"] * max_level_len + ["|"] + ["-"] * max_freq_len + ["|"]
        )

        return (
            f"|{self.name}{name_padding}|{title}{value_padding}|\n"
            f"{horizontal_line}\n"
            f"{rows}"
            f"|{total_title}{total_title_padding}|{self.total/norm}{total_padding}|"
        )

    def __add__(self, that):
        """Combines two FrequencyTable and return
        a new one. All the frequencies are sum together.
        This is not a mathematical sum.
        """
        this_copy = self._counter.copy()
        for key in that.keys():
            if key in this_copy:
                this_copy[key] += that[key]
            else:
                this_copy[key] = that[key]

        return FrequencyTable(this_copy)

    def __str__(self):
        return f"Frequency table (rv:'{self.discrete_rv.name}', total:{self.total})"

    def __repr__(self):
        return self.__str__()

    def __mul__(self, that):
        return self.product(that)

    def __rmul__(self, that):
        # Always rely on the left-multiplication
        return that.__mul__(self)


class DiscreteDistribution(Distribution):
    """A discrete distribution for one or more random variables (RV).
    When the provided class labels are tuples, it takes each position
    in the tuple as a random variable and find its levels.
    """

    def __init__(self, samples, names=None):
        """Construct a DiscreteDistribution from the number of occurenc in samples.

        Args:
            samples (iterable): An iterable that contains the observed samples
                                or a dictionary of (key:frequency).
            names (list, optional): List of names of the random variables.
                                    If it is not provided, it creates as 'Xn'.
                                    Defaults to None.
        """
        super().__init__(samples)
        self.rvs = MultiDiscreteRV(list(self.keys()), names)
        self.names = self.rvs.names

    @classmethod
    def from_np_array(cls, samples, names=None):
        """Construct a DiscreteDistribution from a 2d numpy array or list of lists.
           The resulting keys are tuples.

        Args:
            samples (list or numpy.ndarray): the observed samples.
            names (list, optional): List of names of the random variables.
                                    If it is not provided, it creates as 'Xn'.
                                    Defaults to None.

        Raises:
            ValueError: Raises when the samples argument is not list or
                        numpy.ndarray.
        """
        if not isinstance(samples, (np.ndarray, list)):
            raise ValueError(
                "'sample' argument must be numpy 2D ndarray or list of list."
            )
        elif isinstance(samples, list) and not isinstance(samples[0], list):
            raise ValueError(
                "'sample' argument must be numpy 2D ndarray or list of list."
            )
        # Convert rows to element, before calling
        # the construct
        return cls(samples=[tuple(row) for row in samples], names=names)

    @classmethod
    def genfromtxt(cls, fname, dtypes, names=None, **kwargs):
        """Calls numpy genfromtxt to load the data.
           The resulting keys are tuples.

        Args:
            fname (file, str, pathlib.Path): file name
            dtypes (dtype, optional): Data type of the resulting array.
                   If None, the dtypes will be determined by the contents
                   of each column, individually.
            names (list, optional): List of names of the
                                    random variables.
                                    If it is not provided, it creates as 'Xn'.
                                    Defaults to None.

        """
        samples = np.genfromtxt(fname, dtypes, **kwargs)
        # Convert rows to element, before calling
        # the construct
        return cls(samples=[tuple(row) for row in samples], names=names)

    def __marginal_by_indices(self, by_indices):
        """Marginalize the distribution over a set of random variables.

        Args:
            by_indices (int or list of int): The index/indices of random
                                      variables to sum over.

        Raises:
            ValueError: Raises when the by argument is None.
            TypeError: Raises when the distribution has only one random
                       variable.

        Returns:
            DiscreteDistribution: A marginalised distribution.
        """
        if by_indices is None:
            raise ValueError("The 'by' argument is None.")
        if len(self.rvs) == 1:
            raise TypeError(
                "This is a single random variable distribution and"
                " cannot be marginalised."
            )

        def marginalize_one_var(counter, index):
            """An inner method to marginalize one random variable.

            Args:
                counter (dict or Counter or FrequencyTable): the dist. that
                        must be marginalized
                index (int): The index of the random variable.

            Returns:
                dict: the marginalised dictionary with corrected keys.
            """
            marginal_dist = {}
            for key in counter.keys():
                # Create a new key by removing the given one
                marginal_key = tuple([k for i, k in enumerate(key) if i != index])
                # Make sure the tuples with single elements
                # convert to its element
                if len(marginal_key) == 1:
                    marginal_key = marginal_key[0]

                # Marginalize by summing the same keys
                # Note: turning the key to tuple is safe here,
                # since always len(key) > 1, otherwise we are
                # marginalizing all the random variables.
                if marginal_key in marginal_dist:
                    marginal_dist[marginal_key] += counter[tuple(key)]
                else:
                    marginal_dist[marginal_key] = counter[tuple(key)]

            return marginal_dist

        # Store the indices for marginalization as a numpy array
        rvs_indices = np.sort(np.r_[by_indices])
        # Marginalize the first random variable
        marginal_dist = marginalize_one_var(self, rvs_indices[0])
        # Marginalize the remaining random variables
        # Note that the indices are shifted in each iteration
        for i, position in enumerate(rvs_indices[1:]):
            marginal_dist = marginalize_one_var(marginal_dist, position - i - 1)

        # Find the complement indices of 'provided' list
        complement = np.array([i for i in range(len(self.rvs)) if i not in by_indices])
        return DiscreteDistribution(marginal_dist, names=self.rvs.names[complement])

    def marginal(self, by_names):
        """Marginalize the distribution over a set of random variables.

        Args:
            by_names (list): List of variable names to marginalised.

        Raises:
            ValueError: Raises when one of the random variable names is
                        not defined in rvs.
                        Or raises when requested fo all the random variables.

        Returns:
            DiscreteDistribution: A new marginalised distribution.
        """
        for name in by_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is not defined.")

        if len(by_names) == self.rvs.size:
            raise ValueError("Cannot marginalize on all the random variables.")

        by_indices = [i for i, name in enumerate(self.rvs.names) if name in by_names]
        return self.__marginal_by_indices(by_indices)

    def __reduce_by_indices(self, by_indices):
        """Reduce the distribution by one or more factors.

        Args:
            by_indices (dict): A dictionary that its 'key' is the index
                               of the random variable and its 'value'
                               is the factor that must be reduced by.

        Returns:
            [DiscreteDistribution]: A reduce distribution.
        """

        def single_slice(counter, index, value):
            slice_dist = {}
            for key in counter.keys():
                key_as_tuple = tuple(key)
                if key_as_tuple[index] != value:
                    continue
                # Create a new key by removing the given one
                slice_key = tuple([k for i, k in enumerate(key) if i != index])
                # Make sure the tuples with single elements
                # convert to its element
                if len(slice_key) == 1:
                    slice_key = slice_key[0]
                # keep the value
                if slice_key in slice_dist:
                    slice_dist[slice_key] += counter[tuple(key)]
                else:
                    slice_dist[slice_key] = counter[tuple(key)]
            return slice_dist

        for i, index in enumerate(by_indices):
            value = by_indices[index]
            if i == 0:
                # Slice the first variable
                slice_dist = single_slice(self, index, value)
            else:
                # Marginalize the remaining random variables
                # Note that the indices are shifted in each iteration
                slice_dist = single_slice(slice_dist, index - i, value)

        # Find the complement indices of 'by_indices' list
        complement = np.array([i for i in range(len(self.rvs)) if i not in by_indices])
        return DiscreteDistribution(slice_dist, self.rvs.names[complement])

    def reduce(self, by_names):
        """Reduce the distribution by one or more factors.

        Args:
            by_names (dict): A dictionary that its 'key' is the name
                             of the random variable and its 'value'
                             is the factor that must be reduced by.

        Raises:
            ValueError: If the provided RV names do not exist in the distribution.

        Returns:
            [DiscreteDistribution]: A reduce distribution.
        """
        for name in by_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is" f" not defined.")

        by_indices = {self.rvs.index_of(rv): value for rv, value in by_names.items()}
        return self.__reduce_by_indices(by_indices)

    def __condition_on_by_indices(self, on_names_indices):
        """Creates the conditional distribution based on
           the provided indices of random variables.

        Args:
            on_names_indices (list): List of indices of provided random
                                     variables.

        Raises:
            ValueError: Raises when the provided_indices in None.
            TypeError: Raises when the distribution has only one
                       random variable.

        Returns:
            DiscreteConditionalDistribution
        """

        if on_names_indices is None:
            raise ValueError("The 'provided_indices' argument is None.")
        if len(self.rvs) == 1:
            raise TypeError("This is a single random variable distribution.")

        # Store the indices for marginalization as a numpy array
        rvs_indices = np.sort(np.r_[on_names_indices])
        # Find the complement indices of 'provided_indices' list
        complement = np.array([i for i in range(len(self.rvs)) if i not in rvs_indices])
        # p(x, y, z | a, b) = p(x, y, z , a, b)/ p(a, b)
        #
        # marginal_dist := p(a, b)
        marginal_dist = self.__marginal_by_indices(complement)
        #
        distributions = {}
        for key in marginal_dist.keys():
            key_as_tuple = Key(key)
            by_index = {index: key_as_tuple[i] for i, index in enumerate(rvs_indices)}
            distributions[key] = self.__reduce_by_indices(by_index)
        return ConditionalDistribution(distributions, self.rvs.names[rvs_indices])

    def condition_on(self, on_names):
        """Creates the conditional distribution based on
           the provided names of random variables.

        Args:
            on_names (list): List of names of provided random
                             variables.

        Raises:
            ValueError: If the provided RV names do not exist in the distribution.

        Returns:
            DiscreteConditionalDistribution
        """
        for name in on_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is" f" not defined.")

        if len(on_names) == self.rvs.size:
            raise ValueError("Cannot condition on all the random variables.")

        by_indices = [i for i, name in enumerate(self.rvs.names) if name in on_names]
        return self.__condition_on_by_indices(by_indices)

    def product(self, right):
        """Multiplies a DiscreteDistribution to this one.
            It is the caller responsibility to make sure
            the 'right' argument is a DiscreteDistribution
            class.

        Args:
            right (DiscreteDistribution): the other one.

        Raises:
            ValueError: Raises when the 'right' argument is  Note
                        DiscreteDistribution class.
            ValueError: Raises when the 'right' is the same as the
                        self class.


        Returns:
            DiscreteDistribution: A new DiscreteDistribution instance.
        """
        if not isinstance(right, DiscreteDistribution):
            raise ValueError("The 'right' argument must be a DiscreteDistribution.")

        if right is self:
            raise ValueError("Multiplying DiscreteDistribution with itself.")

        # Find common variables
        # reorder commons based on their order in left_common_indices
        commons = [
            name for name in self.names if name in (set(self.names) & set(right.names))
        ]
        # When there is no common variable, it is just a simple product
        if len(commons) == 0:
            names = np.r_[self.names, right.names]
            return DiscreteDistribution(
                {
                    Key(k1) * Key(k2): v1 * v2
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
                    prodcut_dict[tuple(left_key) + tuple(right_comp)] = (
                        left_value * right_value
                    )
        # names are the combination of [left_names, right_compelements_names]
        combined_names = np.r_[
            [name for name in self.names],
            [name for name in right.names if name not in commons],
        ]
        return DiscreteDistribution(prodcut_dict, combined_names)

    def _get_random_variable_(self):
        return self.rvs

    def summary(self):
        return (
            "Discrete distribution \n"
            f"random variables:'{self.names}'\n"
            f"levels: {self.rvs}\n"
            f"total:{self.total}\n"
            f"normalised:{np.abs(self.total -1) <= 1e-16}\n"
        )

    def to_table(self, normalised=False):

        title = "probability" if normalised else "frequency"
        total_title = "**total**"

        max_levels_len = [
            np.max(
                [len(str(level)) for level in levels]
                + [len(self.names[i]), len(total_title)]
            )
            for i, levels in enumerate(self.rvs.levels)
        ]

        norm = self.total if normalised else 1
        if norm == 0:
            norm = 1

        max_freq_len = np.max(
            [len(str(value / norm)) for _, value in self.items()]
            + [len(str(self.total)), len(title)]
        )

        def padding(max_len):
            def str_padding(value):
                return "".join([" "] * (max_len - len(str(value))))

            return str_padding

        r_padding = padding(max_freq_len)

        rows = ""
        if self.rvs.size == 1:
            l_padding = padding(max_levels_len[0])
            header = (
                f"|{self.names[0]}{l_padding(self.names[0])}|{title}{r_padding(title)}|"
            )
            horizontal_line = "".join(
                ["|"] + ["-"] * max_levels_len[0] + ["|"] + ["-"] * max_freq_len + ["|"]
            )
            for k, value in sorted(self.items()):
                padding = l_padding(k)
                freq_padding = r_padding(value / norm)
                rows += f"|{freq_padding}{k}|{value/norm}{freq_padding}|\n"
        else:
            header = ""
            horizontal_line = ""
            for i, name in enumerate(self.names):
                header += f"|{name}{padding(max_levels_len[i])(name)}"
                horizontal_line += "|" + "".join(["-"] * max_levels_len[i])
            header += "|title"
            horizontal_line += "|" + "".join(["-"] * max_freq_len)

            for k, value in sorted(self.items()):
                key_str = ""
                for i, k_part in enumerate(k):
                    key_str += f"|{padding(max_levels_len[i])(k_part)}{k_part}"
                freq_padding = r_padding(value / norm)
                rows += f"{key_str}|{value/norm}{freq_padding}|\n"

        return f"{header}\n{horizontal_line}|\n{rows}"

    def __lshift__(self, by_indices):
        """marginalization operator"""
        return self.marginal(by_indices)

    def __or__(self, provided):
        """conditional operator"""
        return self.condition_on(provided)

    def __str__(self):
        return (
            f"Discrete Distribution (rvs:'{self.names}',"
            f" normalised:{np.abs(self.total -1) <= 1e-16})"
        )

    def __repr__(self):
        return self.__str__()

    def __mul__(self, right):
        if isinstance(right, Distribution):
            # Convert any other distribution to DiscreteDistribution
            if not isinstance(right, DiscreteDistribution):
                right_dist = DiscreteDistribution(
                    dict(right.items()), names=[right.name]
                )
            else:
                right_dist = right
        else:
            raise ValueError("The 'right' argument must be a DiscreteDistribution.")

        return self.product(right_dist)

    def __rmul__(self, left):
        if isinstance(left, Distribution):
            # Convert any other distribution to DiscreteDistribution
            if not isinstance(left, DiscreteDistribution):
                left_dist = DiscreteDistribution(dict(left.items()), names=[left.name])
            else:
                left_dist = left
        else:
            raise ValueError("The 'right' argument must be a DiscreteDistribution.")

        return left_dist.product(self)
