from collections import Counter
import numpy as np


class DiscreteRV:
    """Store the details of single discrete random variable."""

    def __init__(self, name, levels):
        """Store the details of single discrete random variable.

        Args:
            name (str): The discrete random variable's name.
            levels (set): discrete levels.
        """
        self.name = str(name)
        self.levels = set(levels)
        self.levels_len = len(self.levels)

    def __len__(self):
        return len(self.levels)


class MultiDiscreteRV:
    """Store the details of or more discrete random variables."""

    def __init__(self, features, names=None, variable_name="X"):
        """Store the details of or more discrete random variables.

           it can generate the names of random variables if it is Note
           provided.

        Args:
            features (list): A list of objects or tuples that contains the independent
                             variables' values.
            names (list, optional): A list of random variable names. Defaults to None.
            variable_name (str, optional): The prefix for automatic name generation.
                                           Defaults to "X".

        Raises:
            ValueError: When the length of provided names is not equal to
                        the length of tuples in 'features'.
            ValueError: When the length of tuples in 'features' are not
                        equal.
        """
        # To check the consistency of the tuples, save
        # the length of the first one and check all the others
        first_row = features[0]
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
                f"'Features' has {rv_len} random variables while"
                f"'names' argument has {len(names)}."
            )
        else:
            self.names = np.array(names)
        # Each RV has its own set of levels
        levels = np.array([set() for i in range(rv_len)])
        # We suppose the classes were tuples
        # and each Random Variable (RV) is positioned
        # in a fix place of the n-tuple.
        # Therefore, the levels of the RV can be
        # found by iterating over each tuple's item
        if rv_len > 1:
            # Convert each features line to tuple
            tuples = (tuple(row) for row in features)
            for row in tuples:
                if len(row) != rv_len:
                    raise ValueError("The length of the features is not consistence.")
                for i, level in enumerate(row):
                    levels[i] |= {level}
        else:
            levels[0] = set(features)
        # Store the DiscreteRV as a dictionary
        self.multi_rvs = {
            name: DiscreteRV(name, l) for name, l in zip(self.names, levels)
        }
        # The size of the MultiDiscreteRV is the same
        # as the number RVs or their names
        self.size = len(self.names)

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

    def has_rv(self, name):
        """Check the name of the random variable.

        Args:
            name (str): Name of the random variables.

        Returns:
            [bool]: True or False
        """
        return name in self.multi_rvs

    def __len__(self):
        return len(self.names)

    def all_levels(self):
        """Return a list of all the levels of random variables.
            levels are set.

        Returns:
            list: List of all the levels of random variables.
        """
        return [item.levels for item in self.multi_rvs.values()]

    def index_of(self, name):
        """Finds the index of the random variable from its name.

        Args:
            name (str): Name of the random variable.

        Returns:
            int: Index of the random variable.
        """
        indices = np.argwhere(self.names == name)
        if indices.shape[0] == 0:
            return -1
        else:
            return indices[0][0]


class FrequencyTable:
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
        if samples is None:
            raise ValueError("samples argument is None.")
        # A private Counter object
        self.__counter = Counter(samples)
        # Elements count
        self.total = np.sum([i for i in self.__counter.values()])
        # Random varable's details
        self.discrete_rv = DiscreteRV(name, levels=self.__counter.keys())

    def frequencies(self, normalised=True):
        """A list of frequencies of class occurenc.

        Args:
            normalised (bool, optional): The normalisation flag.
                                         Defaults to True.

        Returns:
            list: A list of floats or integers, depending of normalisation.
        """
        if self.total == 0 and len(self.discrete_rv) != 0:
            return np.zeros(len(self.discrete_rv))
        elif self.total == 0:
            return np.array([])

        if normalised:
            return np.array([v for v in self.__counter.values()]) / self.total
        else:
            return np.array([v for v in self.__counter.values()])

    def probability(self, key):
        """Gets the probability of the random variable, when
           its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object): the value of the random variable.

        Returns:
            float: probability of the random variable.
        """
        if self.total == 0:
            return 0

        return self.__getitem__(key) / self.total

    def frequency(self, key, normalised=False):
        """Gets the frequency of the random variable, when
           its value is 'key'.

           It return zero if the value is not observed.

        Args:
            key (object): the value of the random variable.
            normalised (bool, optional): normalize the return. Defaults to False.

        Returns:
            int or float: frequency or probability of the random variable.
        """
        if self.total == 0:
            return 0

        if normalised:
            return self.__counter[key] / self.total
        else:
            return self.__counter[key]

    def keys(self):
        return self.__counter.keys()

    def items(self):
        return self.__counter.items()

    def np_keys(self):
        return np.array(list(self.__counter.keys()), dtype=np.dtype("O"))

    def tuple_keys(self):
        return [tuple(key) for key in self.__counter.keys()]

    def most_common(self, num: int = None):
        """List the n most common elements and their counts from the most
            common to the least. If n is None, then list all element counts.

        Args:
            num (int, optional): The maximum length of the returned list.
                               Defaults to None.

        Returns:
            list: A list of tuples. The first element of the tuple is a class
                  key and th second one is its count.
        """
        return self.__counter.most_common(num)

    def product(self, right):
        if not isinstance(right, FrequencyTable):
            raise ValueError("The 'right' argument must be a FrequencyTable.")

        if right is self:
            names = [f"{self.discrete_rv.name}1", f"{self.discrete_rv.name}2"]
        else:
            names = [self.discrete_rv.name, right.discrete_rv.name]

        return DiscreteDistribution(
            {
                (tuple(k1) + tuple(k2)): v1 * v2
                for k1, v1 in self.items()
                for k2, v2 in right.items()
            },
            names,
        )

    def __getitem__(self, key):
        """An indexer that returns the count of the class key.

            Returns zero if the 'key' does not exist in samples or
            the samples iterator is empty.


        Args:
            key (object): The key that specifies the class name in samples.

        Returns:
            [float]: The count of occurrence of class 'key'.
                     Returns zero if the 'key' does not exist in samples or
                     the samples iterator is empty.
        """
        if self.total == 0:
            return 0

        return self.__counter[key]

    def __add__(self, that):
        """Combines two FrequencyTable and return
        a new one. All the frequencies are sum together.
        This is not a mathematical sum.
        """
        this_copy = self.__counter.copy()
        for key in that.keys():
            if key in this_copy:
                this_copy[key] += that[key]
            else:
                this_copy[key] = that[key]

        return FrequencyTable(this_copy)

    def __mul__(self, that):
        return self.product(that)


class DiscreteDistribution(FrequencyTable):
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
                                    Defaults to None.
        """
        super().__init__(samples)
        features = list(self.keys())
        self.rvs = MultiDiscreteRV(features, names)
        self.names = self.rvs.names

    def marginal_by_indices(self, by_indices):
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
            raise TypeError("This is a single random variable distribution.")

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

        Returns:
            DiscreteDistribution: A marginalised distribution.
        """
        for name in by_names:
            if not self.rvs.has_rv(name):
                raise ValueError(f"Random variable {name} is" f" not defined.")

        by_indices = [i for i, name in enumerate(self.rvs.names) if name in by_names]
        return self.marginal_by_indices(by_indices)

    def reduce_by_indices(self, by_indices):
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
            if not self.rvs.has_rv(name):
                raise ValueError(f"Random variable {name} is" f" not defined.")

        by_indices = {self.rvs.index_of(rv): value for rv, value in by_names.items()}
        return self.reduce_by_indices(by_indices)

    def conditional_by_indices(self, provided_indices):
        """Creates the conditional distribution based on
           the provided indices of random variables.

        Args:
            provided_indices (list): List of indices of given random
                                     variables.

        Raises:
            ValueError: Raises when the provided_indices in None.
            TypeError: Raises when the distribution has only one
                       random variable.

        Returns:
            DiscreteConditionalDistribution
        """
        if provided_indices is None:
            raise ValueError("The 'provided_indices' argument is None.")
        if len(self.rvs) == 1:
            raise TypeError("This is a single random variable distribution.")

        # Store the indices for marginalization as a numpy array
        rvs_indices = np.sort(np.r_[provided_indices])
        # Find the complement indices of 'provided_indices' list
        complement = np.array([i for i in range(len(self.rvs)) if i not in rvs_indices])
        # p(x, y, z | a, b) = p(x, y, z , a, b)/ p(a, b)
        #
        # marginal_dist := p(a, b)
        marginal_dist = self.marginal_by_indices(complement)
        #
        distributions = {}
        for key in marginal_dist.keys():
            key_as_tuple = tuple(key)
            by_index = {index: key_as_tuple[i] for i, index in enumerate(rvs_indices)}
            distributions[key] = self.reduce_by_indices(by_index)
        return DiscreteConditionalDistribution(
            distributions, self.rvs.names[rvs_indices]
        )

    def conditional(self, provided):
        """Creates the conditional distribution based on
           the provided names of random variables.

        Args:
            provided (list): List of names of given random
                             variables.

        Raises:
            ValueError: If the provided RV names do not exist in the distribution.

        Returns:
            DiscreteConditionalDistribution
        """
        for name in provided:
            if not self.rvs.has_rv(name):
                raise ValueError(f"Random variable {name} is" f" not defined.")

        by_indices = [i for i, name in enumerate(self.rvs.names) if name in provided]
        return self.conditional_by_indices(by_indices)

    def __lshift__(self, by_indices):
        """marginalization operator"""
        return self.marginal(by_indices)

    def __or__(self, provided):
        """conditional operator"""
        return self.conditional(provided)


class DiscreteConditionalDistribution:
    def __init__(self, distributions, conditional_names):
        self.conditional_rvs = MultiDiscreteRV(
            list(distributions.keys()), conditional_names
        )
        # For each conditional levels, we store its equivalent dist.
        self.distributions = distributions

    def probability(self, key, conditional_key):
        if conditional_key not in self.distributions:
            return 0
        else:
            return self.distributions[conditional_key].probability(key)

    def frequency(self, key, conditional_key, normalised=False):
        if conditional_key not in self.distributions:
            return 0
        else:
            return self.distributions[conditional_key].frequency(key, normalised)
