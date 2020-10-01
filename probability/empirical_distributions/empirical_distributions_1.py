from itertools import groupby
from itertools import islice
from itertools import product
from operator import itemgetter
from multiprocessing import Pool
import numpy as np
from probability import Key
from probability import DiscreteRV
from probability import MultiDiscreteRV
from probability import Distribution

from probability.empirical_distributions.empirical_distributions_2 import (
    ConditionalDistribution,
)


def to_dist(iterable):
    return {(k1, k2): v1 * v2 for (k1, v1), (k2, v2) in iterable}


def to_dist(iterable):
    return {(k1, k2): v1 * v2 for (k1, v1), (k2, v2) in iterable}


class FrequencyTable(Distribution):
    """Provides a frequency table from the number of occurenc of
    observed items as dictionary of (key:frequency) or an iterator
    of observed sample.
    """

    def __init__(self, samples, name="X1", check_keys_consistencies=True):
        """Construct a FrequencyTable from the number of occurenc in samples.

        Args:
            samples (iterable):
                An iterable that contains the observed samples
                or a dictionary of (key:frequency).
            name (str):
                Name of the random variable.
            check_keys_consistencies (bool):
                If True, the consistency of the keys (length
                and type) will be checked.
                It is better to be set 'False' for larger datasets
                for the sake of performance.
                Deafault is False.

        Raises:
            ValueError: Raises when the provided sample is None.
        """
        super().__init__(samples)
        # Random varable's details
        try:
            first_row = next(iter(self.keys()))
        except StopIteration:
            first_row = None

        self.discrete_rv = DiscreteRV(name, first_row)
        self.name = name
        #
        if check_keys_consistencies:
            self._check_keys_consistencies_()

    def product(self, right):
        """Multiplies a Distribution to this one.

        Args:
            right ([Distribution]):
                The other Distribution.

        Raises:
            ValueError:
                Raises When the right argument is not a
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
                Key(k1) + Key(k2): v1 * v2
                for k1, v1 in self.items()
                for k2, v2 in right.items()
            },
            names,
        )

    def product_multi_proc(self, right, process=4):
        """Multiplies a Distribution to this one.
           This is a parall version of product

        Args:
            right ([Distribution]):
                The other Distribution.

            process (int):
                Number of processes

        Raises:
            ValueError:
                Raises When the right argument is not a
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
        total_len = len(self.keys()) * len(right.keys())
        main_iter = product(self.items(), right.items())
        slices = [
            islice(
                main_iter,
                (i * total_len) // process,
                ((i + 1) * total_len) // process,
            )
            for i in range(process)
        ]

        with Pool(process) as pool:
            dicts = pool.map(to_dist, slices)
            counter = dicts[0]
            for next_dict in dicts[1:]:
                counter.update(next_dict)
            return DiscreteDistribution(counter, names)

    def _get_random_variable_(self):
        return self.discrete_rv

    def summary(self):
        return (
            "Frequency table \n"
            f"random variable:'{self.discrete_rv.name}'\n"
            f"total:{self.total}"
        )

    def to_table(self, normalised=False, sort=False):

        title = "probability" if normalised else "frequency"
        total_title = "**total**"

        norm = self.total if (normalised and self.total != 0) else 1

        max_level_len = np.max(
            [len(str(level)) for level, _ in self.items()]
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
        if sort:  # sort by values
            items = reversed(sorted(self.items(), key=lambda item: item[1]))
        else:  # sort by keys
            items = sorted(self.items())
        for k, value in items:
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

    def avg(self, operation=None):
        return self.moment(order=1, operation=operation)

    def std(self, operation=None):
        moment_2 = self.moment(order=2, operation=operation)
        avg = self.avg(operation=operation)
        return moment_2 - avg * avg

    def moment(self, order=1, operation=None):
        if not self.discrete_rv.is_numeric and operation is None:
            raise TypeError(
                f"The random variable '{self.discrete_rv.name}' data type"
                " is not numeric."
            )
        if self.total == 0:
            return 0.0
        if operation is not None:
            x_vec = np.fromiter(operation(self.keys_as_list()), dtype=np.float)
        else:
            x_vec = np.fromiter(self._counter.keys(), dtype=np.float)

        prob_vec = np.fromiter(self._counter.values(), dtype=np.float)
        return np.dot(np.power(x_vec, order), prob_vec) / self.total

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

        return FrequencyTable(this_copy, name=self.name)

    def __str__(self):
        return f"Frequency table (rv:'{self.discrete_rv.name}', total:{self.total})"

    __repr__ = __str__

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

    def __init__(self, samples, names=None, check_keys_consistencies=True):
        """Construct a DiscreteDistribution from the number of occurenc in samples.

        Args:
            samples (iterable):
                An iterable that contains the observed samples
                or a dictionary of (key:frequency).
            names (list, optional):
                List of names of the random variables.
                If it is not provided, it creates as 'Xn'.
                Defaults to None.
            check_keys_consistencies (bool):
                If True, the consistency of the keys (length
                and type) will be checked.
                It is better to be set 'False' for larger datasets
                for the sake of performance.
                Deafault is False.
        """
        super().__init__(samples)
        first_row = next(iter(self.keys()))
        self.rvs = MultiDiscreteRV(first_row, names)
        self.names = self.rvs.names
        #
        if check_keys_consistencies:
            self._check_keys_consistencies_()

    @classmethod
    def from_np_array(cls, samples, names=None):
        """Construct a DiscreteDistribution from a 2d numpy array or list of lists.
           The resulting keys are tuples.

        Args:
            samples (list or numpy.ndarray):
                the observed samples.
            names (list, optional):
                List of names of the random variables.
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
        return cls(samples=[Key(row) for row in samples], names=names)

    def marginal(self, *args):
        """Marginalize the distribution over a set of random variables.

        Args:
            args (list):
                List of variable names to marginalised.

        Raises:
            ValueError:
                Raises when one of the random variable names is
                not defined in rvs.
                Or raises when requested fo all the random variables.

        Returns:
            DiscreteDistribution: A new marginalised distribution.
        """
        by_names = args
        for name in by_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is not defined.")

        if len(by_names) == self.rvs.size:
            raise ValueError("Cannot marginalize on all the random variables.")

        indices = [i for i, name in enumerate(self.rvs.names) if name in by_names]
        # Find the indices of compliment random variables (the other ones that
        # are not part of conditioning)
        comp_indices = np.array([i for i in range(len(self.rvs)) if i not in indices])
        # Convert the self._counter's key:value to 2D numpy array
        # the array rows are (random variables, count)
        arr = self._to_2d_array_()
        # filter the compliment random variables by columns
        filtered_arr = np.c_[arr[:, comp_indices], arr[:, -1]]
        # divide the 2d array's rows to a tuple of
        # compliment variables (row[comp_indices])
        # and count row[-1]
        arr_gen = self._split_matrix_(filtered_arr)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of compliment variables (index zero in itemgetter)
        sorted_arr = sorted(arr_gen, key=itemgetter(0))
        # since the values in each 'group' are
        # (compliment variables, count)
        # here we group by 'compliment variables' and sum
        # the count. Then the dictionary of
        # compliment variables:sum_of_counts
        # is an acceptable argument for DiscreteDistribution
        grouped_arr = {
            k: sum([item[1] for item in g])
            for k, g in groupby(sorted_arr, key=itemgetter(0))
        }
        return DiscreteDistribution(grouped_arr, names=self.rvs.names[comp_indices])

    def reduce(self, **kwargs):
        """Reduce the distribution by one or more factors.

        Args:
            kwargs (dict):
                A dictionary that its 'key' is the name
                of the random variable and its 'value'
                is the factor that must be reduced by.

        Raises:
            ValueError:
                If the provided RV names do not exist in the distribution.

        Returns:
            [DiscreteDistribution]: A reduce distribution.
        """
        by_names = kwargs
        for name in by_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is" f" not defined.")

        indices = [self.rvs.index_of(rv) for rv, _ in by_names.items()]
        values = np.array([value for _, value in by_names.items()], dtype=np.object)
        #
        # Convert the self._counter's key:value to 2D numpy array
        # the array rows are (random variables, count)
        arr_counter = self._to_2d_array_()
        # Find the indices of compliment random variables (the other ones that
        # are not part of reduce)
        compliment_indices = [i for i in range(len(self.rvs)) if i not in indices]
        # filter the 2d array rows by provided values of the reduce
        # conditioned_arr is a boolean one, and filtering happens
        # in the second line
        conditioned_arr = np.all(arr_counter[:, indices] == values, axis=1)
        sliced_arr = arr_counter[conditioned_arr, :]
        # filter the 2d array columns (the compliment random variables)
        # plus the count column (which is the last column)
        sliced_arr = sliced_arr[:, compliment_indices + [-1]]
        # divide the 2d array's rows to a tuple of random variables
        # and count
        # So, we make a generator that divide the rows to the tuple of
        # random variables (tuple(row[:-1]) and count (row[-1])
        arr_gen = self._split_matrix_(sliced_arr)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of random variables (index zero in itemgetter)
        sorted_slice_arr = sorted(arr_gen, key=itemgetter(0))
        # group by the filtered random variables (compliment
        # columns) and sum the counts per key
        # Note that the 'itemgetter' read the first index which
        # is the tuple of compliment columns
        slice_dist = {
            k: sum([item[1] for item in g])
            for k, g in groupby(sorted_slice_arr, key=itemgetter(0))
        }
        # Since we have a dictionary of (compliment columns: counts),
        # it is easy to create a DiscreteDistribution.
        # Obviously, the names of these random variables must
        # be the same as compliment columns, which we selected
        # from self.rvs.names
        return DiscreteDistribution(slice_dist, self.rvs.names[compliment_indices])

    def condition_on(self, *args):
        """Creates the conditional distribution based on
           the provided names of random variables.

        Args:
            args (list):
                List of names of provided random
                variables.

        Raises:
            ValueError:
                If the provided RV names do not exist
                in the distribution.

        Returns:
            DiscreteConditionalDistribution
        """
        on_names = args
        for name in on_names:
            if name not in self.rvs:
                raise ValueError(f"Random variable {name} is" f" not defined.")

        if len(on_names) == self.rvs.size:
            raise ValueError("Cannot condition on all the random variables.")

        if len(self.rvs) == 1:
            raise TypeError("This is a single random variable distribution.")

        indices = [i for i, name in enumerate(self.rvs.names) if name in on_names]
        # Find the indices of compliment random variables (the other ones that
        # are not part of conditioning)
        comp_indices = np.array([i for i in range(len(self.rvs)) if i not in indices])
        # Convert the self._counter's key:value to 2D numpy array
        # the array rows are (random variables, count)
        arr = self._to_2d_array_()
        # divide the 2d array's rows to a tuple of random variables,
        # (row[indices]), compliment variables (row[comp_indices])
        # and count row[-1]
        arr_gen = self._split_matrix_(arr, indices)
        # Before calling the groupby, we have to sort the generator
        # by the tuple of random variables (index zero in itemgetter)
        # And since later we will call the group by on group,
        # for each key we do the inner sort too (index one in itemgetter)
        sorted_arr = sorted(arr_gen, key=itemgetter(0, 1))
        # This method convert a group to a DiscreteDistribution

        def make_dist(group):
            # since the values in 'group' argument are
            # (random variables, compliment variables, count)
            # here we group by 'compliment variables' and sum
            # the count.
            # Then the dictionary of compliment variables:sum_of_counts
            # is an acceptable argument for DiscreteDistribution
            grouped_arr_2 = {
                k: sum([item[2] for item in g2])
                for k, g2 in groupby(group, key=itemgetter(1))
            }
            return DiscreteDistribution(grouped_arr_2, self.rvs.names[comp_indices])

        # For each group (belongs a unique values), we create
        # a DiscreteDistribution in a dictionary comprehension
        grouped_arr = {
            k: make_dist(g) for k, g in groupby(sorted_arr, key=itemgetter(0))
        }
        # The above dictionary is the format that ConditionalDistribution
        # accepts as the first argument
        # The conditional names are self.rvs.names[indices]
        return ConditionalDistribution(grouped_arr, self.rvs.names[indices])

    def product(self, right):
        """Multiplies a DiscreteDistribution to this one.
            It is the caller responsibility to make sure
            the 'right' argument is a DiscreteDistribution
            class.

        Args:
            right (DiscreteDistribution):
               The other distribution.

        Raises:
            ValueError:
                Raises when the 'right' argument is not
                DiscreteDistribution class.
            ValueError:
                Raises when the 'right' is the same as the
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
                    Key(k1) + Key(k2): v1 * v2
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
                    prodcut_dict[Key(left_key) + Key(right_comp)] = (
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

    def _split_matrix_(self, array, indices=None):
        """Convert the 2D array of the distribution
           to a generator of 2-elements tuple like
           ((RV_1, RV_2, ..., RV_n), count)

           If the indices is provided, the tuple is
           like
           ((RV_1, ..., RV_n), (RCV_1, ..., RCV_m), count)
           where RV_i is from indices and RCV_j is
           from its compliment

        Args:
            array (numpy ndarray):
                A 2D array of distribution
            indices (list, optional):
                list of indices to filter the column.
                Defaults to None.
        """

        def to_tuple(row):
            """Convert the row to tuple or single value."""
            if row.size == 1:
                return row[0]
            return tuple(row)

        # divide the 2d array's rows to a tuple of
        # random variables (row[indices])
        # and count row[-1]
        if indices is None:
            return ((to_tuple(row[:-1]), row[-1]) for row in array)

        # Find the indices of compliment random variables (the other ones that
        # are not part of conditioning)
        comp_indices = np.array([i for i in range(len(self.rvs)) if i not in indices])
        # divide the 2d array's rows to a tuple of random variables,
        # (row[indices]), compliment variables (row[comp_indices])
        # and count row[-1]
        return (
            (to_tuple(row[indices]), to_tuple(row[comp_indices]), row[-1])
            for row in array
        )

    def summary(self):
        return (
            "Discrete distribution \n"
            f"random variables:'{self.names}'\n"
            f"levels: {self.rvs}\n"
            f"total:{self.total}\n"
            f"normalised:{np.abs(self.total -1) <= 1e-16}\n"
        )

    def to_table(self, normalised=False, sort=False):

        title = "probability" if normalised else "frequency"
        # total_title = "**total**"

        arr = self._to_2d_array_().astype("U")
        arr_len = np.apply_along_axis(lambda row: [len(item) for item in row], 0, arr)
        max_levels_len = np.max(arr_len[:, :-1], axis=1)

        norm = self.total if normalised else 1
        if norm == 0:
            norm = 1

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
        if self.rvs.size == 1:
            l_padding = padding(max_levels_len[0])
            header = (
                f"|{self.names[0]}{l_padding(self.names[0])}|{title}{r_padding(title)}|"
            )
            horizontal_line = "".join(
                ["|"] + ["-"] * max_levels_len[0] + ["|"] + ["-"] * max_freq_len + ["|"]
            )
            for k, value in items:
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

            for k, value in items:
                key_str = ""
                for i, k_part in enumerate(k):
                    key_str += f"|{padding(max_levels_len[i])(k_part)}{k_part}"
                freq_padding = r_padding(value / norm)
                rows += f"{key_str}|{value/norm}{freq_padding}|\n"

        return f"{header}\n{horizontal_line}\n{rows}"

    def avg(self, indices=None):
        return self.moment(order=1, indices=indices)

    def std(self, indices=None):
        moment_2 = self.moment(order=2, indices=indices)
        avg = self.avg(indices)
        return moment_2 - avg * avg

    def moment(self, order=1, indices=None):
        if self.total == 0:
            return 0.0
        matrix = self._to_2d_array_()
        # random variables
        if indices is None:
            x_matrix = matrix[:, :-1]
        else:
            x_matrix = matrix[:, :-1][:, indices]
        # counts
        prob_vec = matrix[:, -1]
        return np.dot(np.power(x_matrix.T, order), prob_vec).T / self.total

    def __lshift__(self, by_indices):
        """marginalization operator"""
        if isinstance(by_indices, tuple):
            return self.marginal(*by_indices)

        return self.marginal(by_indices)

    def __or__(self, provided):
        """conditional operator"""
        if isinstance(provided, tuple):
            return self.condition_on(*provided)

        return self.condition_on(provided)

    def __str__(self):
        return (
            f"Discrete Distribution (rvs:'{self.names}',"
            f" normalised:{np.abs(self.total -1) <= 1e-16})"
        )

    __repr__ = __str__

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

    def __add__(self, that):
        """Combines two FrequencyTable and return
        a new one. All the frequencies are sum together.
        This is not a mathematical sum.
        """
        if not isinstance(that, DiscreteDistribution):
            raise ValueError(
                "DiscreteDistribution can only adds to DiscreteDistribution."
            )

        for name in self.names:
            if name not in that.names:
                raise ValueError(
                    "Two Distributions does not have " "the same random variables."
                )
        this_copy = self._counter.copy()
        for key in that.keys():
            if key in this_copy:
                this_copy[key] += that[key]
            else:
                this_copy[key] = that[key]

        return DiscreteDistribution(this_copy, names=self.names)
